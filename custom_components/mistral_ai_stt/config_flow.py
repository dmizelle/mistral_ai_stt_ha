import logging
from functools import partial
from typing import Any

import httpx
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_URL
from homeassistant.core import HomeAssistant
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

from .const import DEFAULT_STT_MODEL, DOMAIN

_LOGGER = logging.getLogger(__name__)

DATA_SCHEMA = vol.Schema(
    schema={
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_URL): str,
        vol.Optional(CONF_MODEL): str,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    client = await hass.async_add_executor_job(
        partial(
            Mistral,
            api_key=data.get(CONF_API_KEY),
            server_url=data.get(CONF_URL),
        )
    )

    await client.result().models.retrieve_async(
        model_id=data.get(CONF_MODEL, DEFAULT_STT_MODEL),
    )


class MistralAISTTConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_api(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)

            try:
                await validate_input(self.hass, user_input)
            except SDKError as err:
                if err.status_code == 401:
                    errors["base"] = "invalid_auth"
                else:
                    errors["base"] = "unknown"
                    _LOGGER.exception("Unknown SDK Error")
            except httpx.ConnectError as err:
                errors["base"] = "cannot_connect"
                _LOGGER.exception("Cannot connect")
            except Exception:
                errors["base"] = "unknown"
                _LOGGER.exception("Unhandled exception")

        return self.async_show_form(
            step_id="api",
            data_schema=DATA_SCHEMA,
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        return await self.async_step_api()
