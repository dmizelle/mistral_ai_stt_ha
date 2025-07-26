from __future__ import annotations

from collections.abc import AsyncIterable
import io
import logging
import wave
from mistralai import Mistral, File, TranscriptionStreamDone

import voluptuous as vol

from homeassistant.core import HomeAssistant

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.httpx_client import create_async_httpx_client


_LOGGER = logging.getLogger(__name__)


CONF_API_KEY = "api_key"
CONF_API_URL = "api_url"
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_TEMP = "temperature"

DEFAULT_API_URL = "https://api.mistral.ai"
DEFAULT_MODEL = "voxtral-mini-latest"
DEFAULT_PROMPT = ""
DEFAULT_TEMP = 0

SUPPORTED_MODELS = [
    "voxtral-mini-latest",
    "voxtral-mini-2507"
]

SUPPORTED_LANGUAGES = [
    "af",
    "ar",
    "hy",
    "az",
    "be",
    "bs",
    "bg",
    "ca",
    "zh",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "gl",
    "de",
    "el",
    "he",
    "hi",
    "hu",
    "is",
    "id",
    "it",
    "ja",
    "kn",
    "kk",
    "ko",
    "lv",
    "lt",
    "mk",
    "ms",
    "mr",
    "mi",
    "ne",
    "no",
    "fa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sr",
    "sk",
    "sl",
    "es",
    "sw",
    "sv",
    "tl",
    "ta",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
    "cy",
]

MODEL_SCHEMA = vol.In(SUPPORTED_MODELS)

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY): cv.string,
        vol.Optional(CONF_API_URL, default=DEFAULT_API_URL): cv.string,
        vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): MODEL_SCHEMA,
        vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): cv.string,
        vol.Optional(CONF_TEMP, default=DEFAULT_TEMP): cv.positive_int,
    }
)


async def async_get_engine(hass: HomeAssistant, config: dict, **_):
    """
    Set up the Mistral AI STT component.

    Arguments:
        hass: The HomeAssistant instance
        config: The configuration read from HomeAssistant
    """

    api_key: str = config.get(CONF_API_KEY, "")
    api_url: str = config.get(CONF_API_URL, DEFAULT_API_URL)
    model: str = config.get(CONF_MODEL, DEFAULT_MODEL)
    prompt: str = config.get(CONF_PROMPT, DEFAULT_PROMPT)
    temperature: float = config.get(CONF_TEMP, DEFAULT_TEMP)
    return MistralAISTTProvider(hass, api_key, api_url, model, prompt, temperature)


class MistralAISTTProvider(Provider):
    """
    The Mistral AI STT provider.

    Attributes:
        hass: The current HomeAssistant instance
        name: The name of the provider
    """

    def __init__(self, hass: HomeAssistant, api_key: str, api_url: str, model: str, prompt: str, temperature: float) -> None:
        """
        Initialize the Mistral AI Speech-to-Text Provider

        Arguments:
            hass: The HomeAssistant instance
            api_key: The API Key that should be used to authenticate with the inference endpoint
            api_url: The URL to send requests to for transcription
            model: Which model to use for transcription
            prompt: The prompt to send to the inference endpoint
            temperature: The temperature parameter to pass to the model
        """
        self.hass = hass
        """The current HomeAssistant instance"""

        self.name = "Mistral AI STT"
        """The name of the instantiated provider"""

        self._api_key = api_key
        self._api_url = api_url
        self._model = model
        self._prompt = prompt
        self._temperature = temperature
        self._httpx_client = create_async_httpx_client(hass)
        self._client = Mistral(
            api_key = self._api_key,
            server_url=self._api_url,
            async_client=self._httpx_client,
            debug_logger=_LOGGER,
        )

    @property
    def supported_languages(self) -> list[str]:
        """
        Return a list of supported languages.

        Returns:
            list[str]: A list of supported languages
        """
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """
        Return a list of supported formats.

        Returns:
            list[AudioFormats]: A list of supported audio formats
        """
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """
        Return a list of supported codecs.

        Returns:
            list[AudioCodecs]: A list of supported audio codecs
        """
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """
        Return a list of supported bitrates.

        Returns:
            list[AutoBitRates]: A list of supported audio bitrates
        """
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """
        Return a list of supported samplerates.

        Returns:
            list[AudioSampleRates]: A list of supported audio sample rates
        """
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """
        Return a list of supported channels.

        Returns:
            list[AudioChannels]: A list of supported audio channels
        """
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """
        Asynchronously process the audio stream as it comes in

        Arguments:
            metadata: Metadata information about the audio stream
            stream: The bytes of the audio stream
        """

        _LOGGER.debug(
            "Start processing audio stream for language: %s", metadata.language
        )

        # Collect data
        audio_data = b"".join([chunk async for chunk in stream])

        _LOGGER.debug("Audio data size: %d bytes", len(audio_data))

        # Convert audio data to the correct format
        wav_stream = io.BytesIO()

        with wave.open(wav_stream, "wb") as wf:
            wf.setnchannels(metadata.channel)
            wf.setsampwidth(metadata.bit_rate // 8)
            wf.setframerate(metadata.sample_rate)
            wf.writeframes(audio_data)

        _LOGGER.debug(f"Sending request to MistralAI Endpoint")

        try:
            response =  await self._client.audio.transcriptions.stream_async(
                model=self._model,
                file=File(content = wav_stream.getvalue(), file_name = "audio.wav", content_type="audio/wav"),
                language=metadata.language,
                temperature=self._temperature,
            )

            async for chunk in response:
                if isinstance(chunk.data, TranscriptionStreamDone):
                    return SpeechResult(chunk.data.text, SpeechResultState.SUCCESS)

            _LOGGER.error("Error: Speech to text stream never completed")
            return SpeechResult("", SpeechResultState.ERROR)

        except Exception as err:
            _LOGGER.error("Error: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)
