from __future__ import annotations

import io
import logging
import wave
from collections.abc import AsyncIterable

import homeassistant.helpers.config_validation as cv
import voluptuous as vol
from homeassistant.components.stt import (AudioBitRates, AudioChannels,
                                          AudioCodecs, AudioFormats,
                                          AudioSampleRates, Provider,
                                          SpeechMetadata, SpeechResult,
                                          SpeechResultState,
                                          SpeechToTextEntity)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry
from homeassistant.helpers.entity_platform import \
    AddConfigEntryEntitiesCallback
from homeassistant.helpers.httpx_client import create_async_httpx_client
from homeassistant.helpers.typing import DiscoveryInfoType
from mistralai import File, Mistral, TranscriptionStreamDone

from .const import DEFAULT_STT_MODEL, DOMAIN, SUPPORTED_LANGUAGES

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    try:
        client = Mistral(
            api_key=config_entry.data.get(CONF_API_KEY),
            server_url=config_entry.data.get(CONF_URL),
            async_client=create_async_httpx_client(
                hass,
            ),
            debug_logger=_LOGGER,
        )
    except Exception as err:
        _LOGGER.exception("error creating mistral client")
        raise err

    async_add_entities([MistralAISpeechToTextEntity(config_entry, client)])


class MistralAISpeechToTextEntity(SpeechToTextEntity):
    """
    The Mistral AI STT entity.

    Attributes:
        hass: The current HomeAssistant instance
        name: The name of the provider
    """

    def __init__(self, config_entry: ConfigEntry, client: Mistral) -> None:
        """
        Initialize the Mistral AI Speech-to-Text Provider

        Arguments:
            hass: The HomeAssistant instance
            api_key: The API Key that should be used to authenticate with the inference endpoint
            api_url: The URL to send requests to for transcription
            model: Which model to use for transcription
        """

        self._attr_unique_id = f"{config_entry.entry_id}"
        self._attr_name = config_entry.title
        self._attr_device_info = device_registry.DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            manufacturer="Mistral AI",
            model="La Plateforme",
            entry_type=device_registry.DeviceEntryType.SERVICE,
        )
        self._entry = config_entry
        self._client = client
        self._stt_model = config_entry.options.get(CONF_MODEL, DEFAULT_STT_MODEL)

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
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """
        Return a list of supported codecs.

        Returns:
            list[AudioCodecs]: A list of supported audio codecs
        """
        return [AudioCodecs.PCM]

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

        try:
            with wave.open(wav_stream, "wb") as wf:
                wf.setnchannels(metadata.channel)
                wf.setsampwidth(metadata.bit_rate // 8)
                wf.setframerate(metadata.sample_rate)
                wf.writeframes(audio_data)
        except Exception as err:
            _LOGGER.exception("error reading wav stream")
            return SpeechResult("", SpeechResultState.ERROR)

        _LOGGER.debug(f"Sending request to MistralAI Endpoint")

        try:
            response = await self._client.audio.transcriptions.stream_async(
                model=self._stt_model,
                file=File(
                    content=wav_stream.getvalue(),
                    file_name="audio.wav",
                    content_type="audio/wav",
                ),
                language=metadata.language,
                temperature=0.0,
            )

            async for chunk in response:
                if isinstance(chunk.data, TranscriptionStreamDone):
                    return SpeechResult(chunk.data.text, SpeechResultState.SUCCESS)

            _LOGGER.exception("Error: Speech to text stream never completed")
            return SpeechResult("", SpeechResultState.ERROR)

        except Exception as err:
            _LOGGER.exception("Error: %s", err)
            return SpeechResult("", SpeechResultState.ERROR)
