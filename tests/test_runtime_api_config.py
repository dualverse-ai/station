import base64
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

from google.genai import errors as google_genai_errors

from station import constants
from station import file_io_utils
from station import runtime_api_config
from station.llm_connectors.base import (
    BaseLLMConnector,
    LLMCorruptedThoughtSignatureError,
    LLMConnectorError,
    LLMContextOverflowError,
    LLMTransientAPIError,
)
from station.llm_connectors.gemini import GoogleGeminiConnector
from station.llm_connectors.openai import OpenAIConnector


class RetryRefreshConnector(BaseLLMConnector):
    def __init__(self, agent_data_path):
        self.attempt_generations = []
        self.refresh_count = 0
        super().__init__(
            model_name="test-model",
            agent_name="RetryRefreshAgent",
            agent_data_path=agent_data_path,
            max_retries=1,
            retry_delay_seconds=0,
        )

    def sync_state(self) -> None:
        return None

    def _initialize_chat_session(self) -> None:
        return None

    def _load_history_from_file(self):
        return []

    def _append_turn_to_history_file(self, *args, **kwargs):
        return None

    def _send_message_implementation(self, user_prompt, current_tick, attempt_number=0):
        self.attempt_generations.append(self.api_runtime_config_generation)
        if attempt_number == 0:
            runtime_api_config.apply_update({
                "target": "provider",
                "provider": "openai",
                "base_url": "https://retry-refresh.example.test/v1",
            })
            raise LLMTransientAPIError("first attempt failed")
        return "ok", None, {"total_tokens_in_session": 1}

    def _refresh_runtime_api_config_if_changed(self) -> bool:
        current_generation = runtime_api_config.get_generation()
        if current_generation == self.api_runtime_config_generation:
            return False
        self.api_runtime_config_generation = current_generation
        self.refresh_count += 1
        return True

    def get_chat_history(self):
        return []

    def get_current_total_session_tokens(self):
        return 0


class ContextOverflowRetryConnector(BaseLLMConnector):
    def __init__(self, agent_data_path, fail_count, max_retries=10):
        self.fail_count = fail_count
        self.attempt_numbers = []
        super().__init__(
            model_name="test-model",
            agent_name="ContextOverflowAgent",
            agent_data_path=agent_data_path,
            max_retries=max_retries,
            retry_delay_seconds=0,
        )

    def sync_state(self) -> None:
        return None

    def _initialize_chat_session(self) -> None:
        return None

    def _load_history_from_file(self):
        return []

    def _append_turn_to_history_file(self, *args, **kwargs):
        return None

    def _send_message_implementation(self, user_prompt, current_tick, attempt_number=0):
        self.attempt_numbers.append(attempt_number)
        if len(self.attempt_numbers) <= self.fail_count:
            raise LLMContextOverflowError("input exceeds context")
        return "ok", None, {"total_tokens_in_session": 1}

    def get_chat_history(self):
        return []

    def get_current_total_session_tokens(self):
        return 0


class ProviderFallbackConnector(BaseLLMConnector):
    def __init__(
        self,
        agent_data_path,
        error_cls=LLMTransientAPIError,
        fail_first=True,
        probe_success=False,
        always_fail=False,
        failures_before_success: Optional[int] = None,
        max_retries=1,
        retry_delay_seconds=0,
    ):
        self.api_runtime_provider_id = "openai"
        self.api_runtime_env_names = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
        self._api_runtime_config_snapshot = runtime_api_config.get_config_snapshot(
            self.api_runtime_env_names,
            provider_id="openai",
        )
        self.error_cls = error_cls
        self.fail_first = fail_first
        self.probe_success = probe_success
        self.always_fail = always_fail
        self.failures_before_success = failures_before_success
        self.endpoint_indices = []
        self.applied_indices = []
        self.probe_calls = []
        super().__init__(
            model_name="test-model",
            agent_name="ProviderFallbackAgent",
            agent_data_path=agent_data_path,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )

    def sync_state(self) -> None:
        return None

    def _initialize_chat_session(self) -> None:
        return None

    def _load_history_from_file(self):
        return []

    def _append_turn_to_history_file(self, *args, **kwargs):
        return None

    def _apply_provider_runtime_snapshot(self, snapshot):
        self._api_runtime_config_snapshot = snapshot
        self.api_runtime_config_generation = int(snapshot.get("generation", 0))
        endpoint = snapshot.get("provider_endpoint") or {}
        self.applied_indices.append(endpoint.get("index"))

    def _run_provider_base_recovery_probe(self, snapshot):
        endpoint = snapshot.get("provider_endpoint") or {}
        self.probe_calls.append((self.model_name, endpoint.get("index")))
        return self.probe_success

    def _send_message_implementation(self, user_prompt, current_tick, attempt_number=0):
        self.endpoint_indices.append(self._current_provider_endpoint_index())
        if self.always_fail:
            raise self.error_cls("first endpoint failed")
        if self.failures_before_success is not None and len(self.endpoint_indices) <= self.failures_before_success:
            raise self.error_cls("first endpoint failed")
        if self.fail_first and attempt_number == 0:
            raise self.error_cls("first endpoint failed")
        return "ok", None, {"total_tokens_in_session": 1}

    def get_chat_history(self):
        return []

    def get_current_total_session_tokens(self):
        return 0


class FailingOpenAIResponses:
    def __init__(self, error: Optional[Exception] = None):
        self.calls = []
        self.error = error or RuntimeError("responses endpoint failed")

    def create(self, **kwargs):
        self.calls.append(kwargs)
        raise self.error


class FakeOpenAIStatusError(Exception):
    def __init__(self, status_code: int, message: str = "provider failed"):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = "req-test"
        self.body = {"code": "model_not_found", "message": message}


def make_openai_connector_for_internal_fallback_tests(error: Optional[Exception] = None):
    connector = OpenAIConnector.__new__(OpenAIConnector)
    connector.agent_name = "OpenAIInternalFallbackAgent"
    connector.model_name = "gpt-5.5"
    connector.chat_history = []
    connector.client = type("FakeOpenAIClient", (), {"responses": FailingOpenAIResponses(error)})()
    connector.max_output_tokens = None
    connector.verbosity = None
    connector.prompt_cache_retention = None
    connector.reasoning_effort = "low"
    connector.temperature = 1.0
    connector.system_prompt = None
    connector._debug_api_enabled = lambda: False
    connector.api_runtime_config_generation = runtime_api_config.get_generation()
    connector._api_runtime_config_snapshot = {
        "generation": runtime_api_config.get_generation(),
        "http_proxy": "http://provider-proxy.example.test:8080",
        "https_proxy": None,
        "env": {},
    }
    connector._refresh_count = 0

    def fake_refresh():
        connector._refresh_count += 1
        return False

    connector._refresh_runtime_api_config_if_changed = fake_refresh
    return connector


class RuntimeApiConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_runtime_api_config_", dir="/tmp")
        self.saved_constants = {
            "BASE_STATION_DATA_PATH": constants.BASE_STATION_DATA_PATH,
            "LLM_HTTP_PROXY": constants.LLM_HTTP_PROXY,
            "LLM_HTTPS_PROXY": constants.LLM_HTTPS_PROXY,
            "RESEARCH_CENTER_ENABLED": constants.RESEARCH_CENTER_ENABLED,
            "RESEARCH_CODER_BACKEND": constants.RESEARCH_CODER_BACKEND,
            "EXTERNAL_COUNTER_ENABLED": constants.EXTERNAL_COUNTER_ENABLED,
            "AUTO_EVAL_EXTERNAL_REPORT": constants.AUTO_EVAL_EXTERNAL_REPORT,
        }
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        constants.LLM_HTTP_PROXY = None
        constants.LLM_HTTPS_PROXY = None
        constants.RESEARCH_CENTER_ENABLED = True
        constants.RESEARCH_CODER_BACKEND = "codex"
        constants.EXTERNAL_COUNTER_ENABLED = False
        constants.AUTO_EVAL_EXTERNAL_REPORT = True
        with runtime_api_config._LOCK:
            runtime_api_config._GENERATION = 0
            runtime_api_config._PROVIDER_FALLBACK_STATE.clear()

    def tearDown(self):
        for key, value in self.saved_constants.items():
            setattr(constants, key, value)
        with runtime_api_config._LOCK:
            runtime_api_config._GENERATION = 0
            runtime_api_config._PROVIDER_FALLBACK_STATE.clear()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mask_secret_shows_prefix_only(self):
        masked = runtime_api_config.mask_secret("sk-abcdefghijklmnop")
        self.assertTrue(masked["present"])
        self.assertEqual("sk-abcde", masked["visible_prefix"])
        self.assertEqual("sk-abcde...", masked["masked"])
        self.assertNotIn("fghijklmnop", masked["masked"])

    def test_default_provider_update_sets_env_without_file_persistence(self):
        with patch.dict(os.environ, {}, clear=True):
            result = runtime_api_config.apply_update({
                "target": "provider",
                "provider": "openai",
                "base_url": "https://api.example.test/v1",
                "api_key": "sk-test",
            })

            self.assertEqual("https://api.example.test/v1", os.environ.get("OPENAI_BASE_URL"))
            self.assertEqual("sk-test", os.environ.get("OPENAI_API_KEY"))
            self.assertEqual(1, result["generation"])
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "api_runtime_config.yaml")))

            runtime_api_config.apply_update({
                "target": "provider",
                "provider": "openai",
                "base_url": "",
            })

            self.assertNotIn("OPENAI_BASE_URL", os.environ)
            self.assertEqual("sk-test", os.environ.get("OPENAI_API_KEY"))

    def test_provider_proxy_override_falls_back_to_station_proxy_when_blank(self):
        constants.LLM_HTTP_PROXY = "http://base-proxy.test:8080"
        constants.LLM_HTTPS_PROXY = "http://base-proxy.test:8080"
        with patch.dict(os.environ, {}, clear=True):
            runtime_api_config.apply_update({
                "target": "provider",
                "provider": "openai",
                "http_proxy": "http://openai-proxy.test:8080",
                "https_proxy": "http://openai-proxy.test:8080",
            })

            self.assertEqual("http://openai-proxy.test:8080", os.environ.get("OPENAI_HTTP_PROXY"))
            snapshot = runtime_api_config.get_config_snapshot(provider_id="openai")
            self.assertEqual("http://openai-proxy.test:8080", snapshot["http_proxy"])

            public_config = runtime_api_config.apply_update({
                "target": "provider",
                "provider": "openai",
                "http_proxy": "",
                "https_proxy": "",
            })

            self.assertNotIn("OPENAI_HTTP_PROXY", os.environ)
            snapshot = runtime_api_config.get_config_snapshot(provider_id="openai")
            self.assertEqual("http://base-proxy.test:8080", snapshot["http_proxy"])
            openai_public = next(p for p in public_config["providers"] if p["id"] == "openai")
            self.assertEqual("OPENAI_HTTP_PROXY", openai_public["http_proxy_env"])

    def test_provider_backup_env_cycles_to_next_endpoint(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "OPENAI_BASE_URL": "https://base.example.test/v1",
            "BACKUP_OPENAI_API_KEY": "backup-key;official-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1;",
        }, clear=True):
            base_snapshot = runtime_api_config.get_config_snapshot(
                ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                provider_id="openai",
            )
            self.assertEqual("base-key", base_snapshot["env"]["OPENAI_API_KEY"])
            self.assertEqual(0, base_snapshot["provider_endpoint"]["index"])
            self.assertTrue(base_snapshot["provider_endpoint"]["configured"])

            retry_snapshot = runtime_api_config.record_provider_failure_and_get_retry_snapshot(
                "openai",
                0,
                ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
            )

            self.assertIsNotNone(retry_snapshot)
            self.assertEqual("backup-key", retry_snapshot["env"]["OPENAI_API_KEY"])
            self.assertEqual("https://backup.example.test/v1", retry_snapshot["env"]["OPENAI_BASE_URL"])
            self.assertEqual(1, retry_snapshot["provider_endpoint"]["index"])

    def test_backup_endpoint_blank_proxy_falls_back_to_station_proxy(self):
        constants.LLM_HTTP_PROXY = "http://station-proxy.example.test:8080"
        constants.LLM_HTTPS_PROXY = "http://station-proxy.example.test:8080"
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key;other-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1;https://other.example.test/v1",
            "BACKUP_OPENAI_HTTP_PROXY": ";http://explicit-proxy.example.test:8080",
            "BACKUP_OPENAI_HTTPS_PROXY": ";",
        }, clear=True):
            first_backup = runtime_api_config.get_config_snapshot(
                ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY"],
                provider_id="openai",
                endpoint_index=1,
            )
            second_backup = runtime_api_config.get_config_snapshot(
                ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY"],
                provider_id="openai",
                endpoint_index=2,
            )

            self.assertEqual("http://station-proxy.example.test:8080", first_backup["http_proxy"])
            self.assertEqual("http://station-proxy.example.test:8080", first_backup["https_proxy"])
            self.assertEqual("http://explicit-proxy.example.test:8080", second_backup["http_proxy"])
            self.assertEqual("http://station-proxy.example.test:8080", second_backup["https_proxy"])

    def test_provider_default_promotes_after_rolling_failures(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            for _ in range(10):
                runtime_api_config.record_provider_failure_and_get_retry_snapshot(
                    "openai",
                    0,
                    ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                )

            promoted_snapshot = runtime_api_config.get_config_snapshot(
                ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                provider_id="openai",
            )
            self.assertEqual(1, promoted_snapshot["provider_endpoint"]["index"])
            self.assertEqual("backup-key", promoted_snapshot["env"]["OPENAI_API_KEY"])

    def test_provider_default_ignores_failure_samples_older_than_one_hour(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            with patch("station.runtime_api_config.time.time", return_value=1000.0):
                for _ in range(9):
                    runtime_api_config.record_provider_failure_and_get_retry_snapshot(
                        "openai",
                        0,
                        ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                    )

            with patch("station.runtime_api_config.time.time", return_value=4601.0):
                runtime_api_config.record_provider_failure_and_get_retry_snapshot(
                    "openai",
                    0,
                    ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                )
                snapshot = runtime_api_config.get_config_snapshot(
                    ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                    provider_id="openai",
                )

            self.assertEqual(0, snapshot["provider_endpoint"]["index"])
            self.assertEqual("base-key", snapshot["env"]["OPENAI_API_KEY"])

    def test_invalid_backup_env_raises(self):
        with patch.dict(os.environ, {
            "BACKUP_OPENAI_API_KEY": "backup-key;other-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            with self.assertRaisesRegex(ValueError, "BACKUP_OPENAI_BASE_URL"):
                runtime_api_config.validate_provider_backup_env_config()

            with self.assertRaisesRegex(ValueError, "Invalid provider backup API configuration"):
                runtime_api_config.get_config_snapshot(
                    ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                    provider_id="openai",
                )

    def test_provider_update_accepts_backup_rows(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "base-key"}, clear=True):
            result = runtime_api_config.apply_update({
                "target": "provider",
                "provider": "openai",
                "backup_endpoints": [
                    {
                        "api_key": "backup-key",
                        "base_url": "https://backup.example.test/v1",
                        "http_proxy": "",
                        "https_proxy": "",
                    },
                    {
                        "api_key": "official-key",
                        "base_url": "",
                        "http_proxy": "http://proxy.example.test:8080",
                        "https_proxy": "",
                    },
                ],
            })

            self.assertEqual("backup-key;official-key", os.environ.get("BACKUP_OPENAI_API_KEY"))
            self.assertEqual("https://backup.example.test/v1;", os.environ.get("BACKUP_OPENAI_BASE_URL"))
            self.assertEqual(";http://proxy.example.test:8080", os.environ.get("BACKUP_OPENAI_HTTP_PROXY"))
            openai_public = next(provider for provider in result["providers"] if provider["id"] == "openai")
            self.assertEqual(2, openai_public["backup_endpoint_count"])

    def test_gemini_blank_backup_base_url_bypasses_base_env_override(self):
        connector = object.__new__(GoogleGeminiConnector)
        connector.agent_name = "GeminiRuntimeConfigTest"
        saved_timeout = constants.GEMINI_TIMEOUT
        constants.GEMINI_TIMEOUT = None
        try:
            with patch.dict(os.environ, {
                "GOOGLE_GEMINI_BASE_URL": "https://base-proxy.example.test",
            }, clear=True):
                http_options = connector._build_http_options(None)
        finally:
            constants.GEMINI_TIMEOUT = saved_timeout

        self.assertIsNotNone(http_options)
        self.assertEqual(
            "https://generativelanguage.googleapis.com",
            http_options.base_url.rstrip("/"),
        )

    def test_gemini_client_error_details_include_raw_sdk_fields(self):
        connector = object.__new__(GoogleGeminiConnector)
        error = google_genai_errors.ClientError(
            401,
            {
                "error": {
                    "code": 401,
                    "message": "",
                    "status": "UNAUTHENTICATED",
                    "details": [{"reason": "API_KEY_INVALID"}],
                }
            },
            response=None,
        )

        details = connector._format_gemini_error_details(error)

        self.assertIn("type=ClientError", details)
        self.assertIn("code=401", details)
        self.assertIn("status='UNAUTHENTICATED'", details)
        self.assertIn("API_KEY_INVALID", details)

    def test_gemini_corrupted_thought_signature_detection_matches_custom_and_official_errors(self):
        connector = object.__new__(GoogleGeminiConnector)

        self.assertTrue(connector._is_corrupted_thought_signature_error(
            "400 INVALID_ARGUMENT. {'error': {'message': 'Corrupted thought signature.'}}"
        ))
        self.assertTrue(connector._is_corrupted_thought_signature_error(
            "400 None. {'error': {'message': 'Corrupted thought signature.', 'type': 'upstream_error'}}"
        ))
        self.assertTrue(connector._is_corrupted_thought_signature_error(
            "400 None. {'error': {'message': 'corrupted_thought_signature', 'type': 'upstream_error'}}"
        ))
        custom_base_error = (
            "400 None. {'error': {'message': '上游请求参数无效，请检查后重试 "
            "(cch_session_id: sess_mp28re87_24f716ebeea3)', "
            "'type': 'invalid_request_error', 'param': '', 'code': 'invalid_request_error'}}"
        )
        self.assertFalse(connector._is_corrupted_thought_signature_error(custom_base_error))
        self.assertFalse(connector._is_corrupted_thought_signature_error(
            "400 INVALID_ARGUMENT. {'error': {'message': 'input token count exceeds the maximum number of tokens allowed'}}"
        ))

    def test_gemini_token_count_does_not_use_completion_fallback_when_count_tokens_is_unavailable(self):
        class FakeModels:
            def __init__(self):
                self.calls = []

            def count_tokens(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(total_tokens=None)

        class FakeChats:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                raise AssertionError("Gemini token counting should not use completion fallback")

        connector = object.__new__(GoogleGeminiConnector)
        connector.agent_name = "GeminiTokenCountFallbackTest"
        connector.model_name = "gemini-test"
        connector.active_base_url = "https://generativelanguage.googleapis.com"
        connector.client = SimpleNamespace(models=FakeModels(), chats=FakeChats())
        connector.chat_session = SimpleNamespace(get_history=lambda: [
            SimpleNamespace(role="user", parts=[SimpleNamespace(text="hello")]),
        ])
        connector._debug_api_enabled = lambda: False

        token_count = connector.get_current_total_session_tokens()

        self.assertIsNone(token_count)
        self.assertEqual(1, len(connector.client.models.calls))
        self.assertEqual([], connector.client.chats.calls)

    def test_gemini_invalid_signature_detection_uses_protobuf_envelope(self):
        connector = object.__new__(GoogleGeminiConnector)
        connector.agent_name = "GeminiSignatureDetectionTest"
        good_signature = base64.b64encode(b"\x0a\x03abc").decode("ascii")
        nested_good_signature = base64.b64encode(b"\x12\x05\x0a\x03abc").decode("ascii")
        first_bad_signature = base64.b64encode(b"\x01\x8f\x3dk-first").decode("ascii")
        last_bad_signature = base64.b64encode(b"\x01\x8f\x3dk-last").decode("ascii")

        self.assertFalse(connector._thought_signature_diagnostics(good_signature)["invalid_signature"])
        self.assertFalse(connector._thought_signature_diagnostics(nested_good_signature)["invalid_signature"])

        results = connector._find_invalid_thought_signatures([
            {"role": "model", "thought_signature": good_signature},
            {"role": "model", "thought_signature": nested_good_signature},
            {"role": "model", "thought_signature": first_bad_signature},
            {"role": "user", "thought_signature": last_bad_signature},
            {"role": "model", "thought_signature": last_bad_signature},
        ])

        self.assertEqual(2, len(results))
        result = results[-1]
        self.assertIsNotNone(result)
        self.assertEqual(last_bad_signature, result["signature"])
        self.assertEqual(4, result["processed_index"])
        self.assertTrue(result["invalid_signature"])
        self.assertTrue(result["wrong_leading_byte"])
        self.assertTrue(result["repairable_raw_payload"])
        self.assertIsNotNone(result["repaired_signature"])
        self.assertTrue(result["first_bytes_hex"].startswith("01 8f 3d 6b"))

    def test_gemini_removes_invalid_signature_from_persistent_history(self):
        connector = object.__new__(GoogleGeminiConnector)
        connector.agent_name = "GeminiSignatureRemovalTest"
        connector.history_file_path = os.path.join(self.tmpdir, "agents", "GeminiSignatureRemovalTest", "llm_chat_history.yamll")
        good_signature = base64.b64encode(b"\x0a\x03abc").decode("ascii")
        bad_signature = base64.b64encode(b"\x01\x8f\x3dk-bad").decode("ascii")

        file_io_utils.append_yaml_line({
            "tick": 1,
            "role": "model",
            "parts": [{"text": "good", "thought_signature": good_signature}],
        }, connector.history_file_path)
        file_io_utils.append_yaml_line({
            "tick": 2,
            "role": "model",
            "parts": [{"text": "bad", "thought_signature": bad_signature}],
        }, connector.history_file_path)

        removed = connector._remove_persistent_thought_signature({
            "signature": bad_signature,
            "signature_b64_length": len(bad_signature),
            "decoded_length": len(base64.b64decode(bad_signature)),
            "first_bytes_hex": "01 8f 3d 6b",
            "invalid_signature": True,
            "wrong_leading_byte": True,
            "reason": "invalid_protobuf_envelope=ValueError: invalid_field_number_0_at_offset=0",
        })

        entries = file_io_utils.load_yaml_lines(connector.history_file_path)
        self.assertTrue(removed)
        self.assertEqual(good_signature, entries[0]["parts"][0]["thought_signature"])
        self.assertNotIn("thought_signature", entries[1]["parts"][0])
        self.assertEqual("bad", entries[1]["parts"][0]["text"])

    def test_gemini_corrupted_signature_retry_repairs_raw_payload_and_persists_cleanup(self):
        class FakeGeminiChat:
            def __init__(self, fail_with_corrupted_signature=False):
                self.fail_with_corrupted_signature = fail_with_corrupted_signature

            def send_message(self, _prompt, config=None):
                if self.fail_with_corrupted_signature:
                    raise google_genai_errors.ClientError(
                        400,
                        {"error": {"message": "Corrupted thought signature."}},
                        response=None,
                    )
                return SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            content=SimpleNamespace(
                                parts=[SimpleNamespace(text="ok", thought=False)]
                            )
                        )
                    ],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=1,
                        candidates_token_count=1,
                        cached_content_token_count=None,
                        thoughts_token_count=None,
                        total_token_count=2,
                    ),
                )

            def get_history(self):
                return []

        class FakeGeminiChats:
            def __init__(self):
                self.created_histories = []

            def create(self, model, history=None):
                self.created_histories.append(history)
                return FakeGeminiChat()

        connector = object.__new__(GoogleGeminiConnector)
        connector.agent_name = "GeminiSignatureRecoveryTest"
        connector.model_name = "gemini-test"
        connector.agent_prune_blocks = []
        connector.history_file_path = os.path.join(self.tmpdir, "agents", "GeminiSignatureRecoveryTest", "llm_chat_history.yamll")
        connector.client = SimpleNamespace(chats=FakeGeminiChats())
        connector.chat_session = FakeGeminiChat(fail_with_corrupted_signature=True)
        connector.generation_config = None
        connector.active_endpoint_name = "base"
        connector.active_base_url = "https://api.example.test"
        connector.persist_to_disk = True
        connector._last_api_metadata = None
        connector._last_model_turn_thought_signature = None
        connector._debug_api_enabled = lambda: False

        good_signature = base64.b64encode(b"\x0a\x03abc").decode("ascii")
        bad_signature = base64.b64encode(b"\x01\x8f\x3dk-bad").decode("ascii")
        repaired_signature = connector._repair_raw_payload_thought_signature(base64.b64decode(bad_signature))
        file_io_utils.append_yaml_line({"tick": 1, "role": "user", "parts": [{"text": "u1"}]}, connector.history_file_path)
        file_io_utils.append_yaml_line({
            "tick": 1,
            "role": "model",
            "parts": [{"text": "good", "thought_signature": good_signature}],
        }, connector.history_file_path)
        file_io_utils.append_yaml_line({"tick": 2, "role": "user", "parts": [{"text": "u2"}]}, connector.history_file_path)
        file_io_utils.append_yaml_line({
            "tick": 2,
            "role": "model",
            "parts": [{"text": "bad", "thought_signature": bad_signature}],
        }, connector.history_file_path)

        response, thinking, token_info = connector._send_message_implementation("prompt", current_tick=3)

        self.assertEqual("ok", response)
        self.assertIsNone(thinking)
        self.assertEqual(2, token_info["total_tokens_in_session"])
        self.assertGreaterEqual(len(connector.client.chats.created_histories), 2)
        retry_history_repr = repr(connector.client.chats.created_histories[0])
        self.assertIn(good_signature, retry_history_repr)
        self.assertIn(repaired_signature, retry_history_repr)
        self.assertNotIn(bad_signature, retry_history_repr)

        entries = file_io_utils.load_yaml_lines(connector.history_file_path)
        serialized_entries = repr(entries)
        self.assertIn(good_signature, serialized_entries)
        self.assertIn(repaired_signature, serialized_entries)
        self.assertNotIn(bad_signature, serialized_entries)
        self.assertEqual("prompt", entries[-2]["parts"][0]["text"])
        self.assertEqual("ok", entries[-1]["parts"][0]["text"])

    def test_gemini_corrupted_signature_retry_removes_bad_signature_if_repair_fails(self):
        class FakeGeminiChat:
            def __init__(self, fail_with_corrupted_signature=False):
                self.fail_with_corrupted_signature = fail_with_corrupted_signature

            def send_message(self, _prompt, config=None):
                if self.fail_with_corrupted_signature:
                    raise google_genai_errors.ClientError(
                        400,
                        {"error": {"message": "Corrupted thought signature."}},
                        response=None,
                    )
                return SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            content=SimpleNamespace(
                                parts=[SimpleNamespace(text="ok", thought=False)]
                            )
                        )
                    ],
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=1,
                        candidates_token_count=1,
                        cached_content_token_count=None,
                        thoughts_token_count=None,
                        total_token_count=2,
                    ),
                )

            def get_history(self):
                return []

        class FakeGeminiChats:
            def __init__(self):
                self.created_histories = []
                self.failures = [True, False]

            def create(self, model, history=None):
                self.created_histories.append(history)
                should_fail = self.failures.pop(0) if self.failures else False
                return FakeGeminiChat(fail_with_corrupted_signature=should_fail)

        connector = object.__new__(GoogleGeminiConnector)
        connector.agent_name = "GeminiSignatureRemovalFallbackTest"
        connector.model_name = "gemini-test"
        connector.agent_prune_blocks = []
        connector.history_file_path = os.path.join(self.tmpdir, "agents", "GeminiSignatureRemovalFallbackTest", "llm_chat_history.yamll")
        connector.client = SimpleNamespace(chats=FakeGeminiChats())
        connector.chat_session = FakeGeminiChat(fail_with_corrupted_signature=True)
        connector.generation_config = None
        connector.active_endpoint_name = "base"
        connector.active_base_url = "https://api.example.test"
        connector.persist_to_disk = True
        connector._last_api_metadata = None
        connector._last_model_turn_thought_signature = None
        connector._debug_api_enabled = lambda: False

        good_signature = base64.b64encode(b"\x0a\x03abc").decode("ascii")
        bad_signature = base64.b64encode(b"\x01\x8f\x3dk-bad").decode("ascii")
        repaired_signature = connector._repair_raw_payload_thought_signature(base64.b64decode(bad_signature))
        file_io_utils.append_yaml_line({"tick": 1, "role": "user", "parts": [{"text": "u1"}]}, connector.history_file_path)
        file_io_utils.append_yaml_line({
            "tick": 1,
            "role": "model",
            "parts": [{"text": "good", "thought_signature": good_signature}],
        }, connector.history_file_path)
        file_io_utils.append_yaml_line({"tick": 2, "role": "user", "parts": [{"text": "u2"}]}, connector.history_file_path)
        file_io_utils.append_yaml_line({
            "tick": 2,
            "role": "model",
            "parts": [{"text": "bad", "thought_signature": bad_signature}],
        }, connector.history_file_path)

        response, thinking, token_info = connector._send_message_implementation("prompt", current_tick=3)

        self.assertEqual("ok", response)
        self.assertIsNone(thinking)
        self.assertEqual(2, token_info["total_tokens_in_session"])
        self.assertGreaterEqual(len(connector.client.chats.created_histories), 3)
        repair_history_repr = repr(connector.client.chats.created_histories[0])
        removal_history_repr = repr(connector.client.chats.created_histories[1])
        self.assertIn(repaired_signature, repair_history_repr)
        self.assertIn(good_signature, removal_history_repr)
        self.assertNotIn(bad_signature, removal_history_repr)

        entries = file_io_utils.load_yaml_lines(connector.history_file_path)
        serialized_entries = repr(entries)
        self.assertIn(good_signature, serialized_entries)
        self.assertNotIn(bad_signature, serialized_entries)
        self.assertNotIn(repaired_signature, serialized_entries)
        self.assertEqual("prompt", entries[-2]["parts"][0]["text"])
        self.assertEqual("ok", entries[-1]["parts"][0]["text"])

    def test_gemini_omission_retry_non_corrupted_error_returns_to_transient_retry(self):
        class FakeGeminiChat:
            def __init__(self, error_mode=None):
                self.error_mode = error_mode

            def send_message(self, _prompt, config=None):
                if self.error_mode == "corrupt":
                    raise google_genai_errors.ClientError(
                        400,
                        {"error": {"message": "Corrupted thought signature."}},
                        response=None,
                    )
                if self.error_mode == "invalid_request":
                    raise google_genai_errors.ClientError(
                        400,
                        {"error": {"message": "upstream rejected the request", "code": "invalid_request_error"}},
                        response=None,
                    )
                return SimpleNamespace(
                    candidates=[
                        SimpleNamespace(
                            content=SimpleNamespace(
                                parts=[SimpleNamespace(text="ok", thought=False)]
                            )
                        )
                    ],
                    usage_metadata=SimpleNamespace(total_token_count=2),
                )

            def get_history(self):
                return []

        class FakeGeminiChats:
            def __init__(self):
                self.created_histories = []
                self.error_modes = ["corrupt", "invalid_request"]

            def create(self, model, history=None):
                self.created_histories.append(history)
                error_mode = self.error_modes.pop(0) if self.error_modes else None
                return FakeGeminiChat(error_mode=error_mode)

        connector = object.__new__(GoogleGeminiConnector)
        connector.agent_name = "GeminiSignatureNonCorruptAfterOmitTest"
        connector.model_name = "gemini-test"
        connector.agent_prune_blocks = []
        connector.history_file_path = os.path.join(self.tmpdir, "agents", "GeminiSignatureNonCorruptAfterOmitTest", "llm_chat_history.yamll")
        connector.client = SimpleNamespace(chats=FakeGeminiChats())
        connector.chat_session = FakeGeminiChat(error_mode="corrupt")
        connector.generation_config = None
        connector.active_endpoint_name = "base"
        connector.active_base_url = "https://api.example.test"
        connector.persist_to_disk = True
        connector._last_api_metadata = None
        connector._last_model_turn_thought_signature = None
        connector._debug_api_enabled = lambda: False

        good_signature = base64.b64encode(b"\x0a\x03abc").decode("ascii")
        bad_signature = base64.b64encode(b"\x01\x8f\x3dk-bad").decode("ascii")
        repaired_signature = connector._repair_raw_payload_thought_signature(base64.b64decode(bad_signature))
        file_io_utils.append_yaml_line({"tick": 1, "role": "user", "parts": [{"text": "u1"}]}, connector.history_file_path)
        file_io_utils.append_yaml_line({
            "tick": 1,
            "role": "model",
            "parts": [{"text": "good", "thought_signature": good_signature}],
        }, connector.history_file_path)
        file_io_utils.append_yaml_line({"tick": 2, "role": "user", "parts": [{"text": "u2"}]}, connector.history_file_path)
        file_io_utils.append_yaml_line({
            "tick": 2,
            "role": "model",
            "parts": [{"text": "bad", "thought_signature": bad_signature}],
        }, connector.history_file_path)

        with self.assertRaises(LLMTransientAPIError) as raised:
            connector._send_message_implementation("prompt", current_tick=3)

        self.assertIn("non-corrupted error", str(raised.exception))
        self.assertGreaterEqual(len(connector.client.chats.created_histories), 3)
        repair_history_repr = repr(connector.client.chats.created_histories[0])
        removal_history_repr = repr(connector.client.chats.created_histories[1])
        self.assertIn(repaired_signature, repair_history_repr)
        self.assertIn(good_signature, removal_history_repr)
        self.assertNotIn(bad_signature, removal_history_repr)

        entries = file_io_utils.load_yaml_lines(connector.history_file_path)
        serialized_entries = repr(entries)
        self.assertIn(good_signature, serialized_entries)
        self.assertNotIn(bad_signature, serialized_entries)
        self.assertNotIn(repaired_signature, serialized_entries)

    def test_gemini_omission_retry_corrupted_error_still_pauses(self):
        class FakeGeminiChat:
            def send_message(self, _prompt, config=None):
                raise google_genai_errors.ClientError(
                    400,
                    {"error": {"message": "Corrupted thought signature."}},
                    response=None,
                )

            def get_history(self):
                return []

        class FakeGeminiChats:
            def __init__(self):
                self.created_histories = []

            def create(self, model, history=None):
                self.created_histories.append(history)
                return FakeGeminiChat()

        connector = object.__new__(GoogleGeminiConnector)
        connector.agent_name = "GeminiSignatureStillCorruptAfterOmitTest"
        connector.model_name = "gemini-test"
        connector.agent_prune_blocks = []
        connector.history_file_path = os.path.join(self.tmpdir, "agents", "GeminiSignatureStillCorruptAfterOmitTest", "llm_chat_history.yamll")
        connector.client = SimpleNamespace(chats=FakeGeminiChats())
        connector.chat_session = FakeGeminiChat()
        connector.generation_config = None
        connector.active_endpoint_name = "base"
        connector.active_base_url = "https://api.example.test"
        connector.persist_to_disk = True
        connector._last_api_metadata = None
        connector._last_model_turn_thought_signature = None
        connector._debug_api_enabled = lambda: False

        good_signature = base64.b64encode(b"\x0a\x03abc").decode("ascii")
        bad_signature = base64.b64encode(b"\x01\x8f\x3dk-bad").decode("ascii")
        file_io_utils.append_yaml_line({"tick": 1, "role": "user", "parts": [{"text": "u1"}]}, connector.history_file_path)
        file_io_utils.append_yaml_line({
            "tick": 1,
            "role": "model",
            "parts": [{"text": "good", "thought_signature": good_signature}],
        }, connector.history_file_path)
        file_io_utils.append_yaml_line({"tick": 2, "role": "user", "parts": [{"text": "u2"}]}, connector.history_file_path)
        file_io_utils.append_yaml_line({
            "tick": 2,
            "role": "model",
            "parts": [{"text": "bad", "thought_signature": bad_signature}],
        }, connector.history_file_path)

        with self.assertRaises(LLMCorruptedThoughtSignatureError):
            connector._send_message_implementation("prompt", current_tick=3)

        entries = file_io_utils.load_yaml_lines(connector.history_file_path)
        self.assertIn(bad_signature, repr(entries))

    def test_openai_blank_backup_base_url_bypasses_base_env_override(self):
        connector = object.__new__(OpenAIConnector)
        connector.agent_name = "OpenAIRuntimeConfigTest"
        connector.api_key = None
        connector._explicit_api_key = False
        snapshot = {
            "env": {
                "OPENAI_API_KEY": "official-key",
                "OPENAI_BASE_URL": "",
            }
        }

        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://base-proxy.example.test/v1",
        }, clear=True):
            client_kwargs = connector._build_client_kwargs_from_runtime_snapshot(snapshot)

        self.assertEqual("official-key", client_kwargs["api_key"])
        self.assertEqual("https://api.openai.com/v1", client_kwargs["base_url"])

    def test_provider_update_preserves_existing_backup_key_when_blank(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key;other-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1;https://other.example.test/v1",
        }, clear=True):
            runtime_api_config.apply_update({
                "target": "provider",
                "provider": "openai",
                "backup_endpoints": [
                    {
                        "existing_index": 1,
                        "api_key": "",
                        "base_url": "https://updated.example.test/v1",
                        "http_proxy": "",
                        "https_proxy": "",
                    },
                ],
            })

            self.assertEqual("backup-key", os.environ.get("BACKUP_OPENAI_API_KEY"))
            self.assertEqual("https://updated.example.test/v1", os.environ.get("BACKUP_OPENAI_BASE_URL"))

    def test_provider_update_rejects_backup_semicolon(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "base-key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "cannot contain ';'"):
                runtime_api_config.apply_update({
                    "target": "provider",
                    "provider": "openai",
                    "backup_endpoints": [
                        {
                            "api_key": "backup-key",
                            "base_url": "https://one.example.test/v1;https://two.example.test/v1",
                        },
                    ],
                })

    def test_provider_update_rejects_invalid_backup_without_partial_base_change(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "OPENAI_BASE_URL": "https://old.example.test/v1",
        }, clear=True):
            with self.assertRaisesRegex(ValueError, "cannot contain ';'"):
                runtime_api_config.apply_update({
                    "target": "provider",
                    "provider": "openai",
                    "base_url": "https://new.example.test/v1",
                    "backup_endpoints": [
                        {
                            "api_key": "backup-key",
                            "base_url": "https://one.example.test/v1;https://two.example.test/v1",
                        },
                    ],
                })

            self.assertEqual("https://old.example.test/v1", os.environ.get("OPENAI_BASE_URL"))
            self.assertNotIn("BACKUP_OPENAI_API_KEY", os.environ)

    def test_no_backup_env_keeps_existing_provider_behavior(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "OPENAI_BASE_URL": "https://base.example.test/v1",
        }, clear=True):
            snapshot = runtime_api_config.get_config_snapshot(
                ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                provider_id="openai",
            )
            self.assertFalse(snapshot["provider_endpoint"]["configured"])
            self.assertEqual("base-key", snapshot["env"]["OPENAI_API_KEY"])

            retry_snapshot = runtime_api_config.record_provider_failure_and_get_retry_snapshot(
                "openai",
                0,
                ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
            )
            self.assertIsNone(retry_snapshot)

    def test_codex_proxy_override_falls_back_to_station_proxy_when_blank(self):
        constants.LLM_HTTP_PROXY = "http://base-proxy.test:8080"
        constants.LLM_HTTPS_PROXY = "http://base-proxy.test:8080"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                "http://base-proxy.test:8080",
                runtime_api_config.get_codex_proxy_values()["http_proxy"],
            )

            runtime_api_config.apply_update({
                "target": "codex",
                "http_proxy": "http://codex-proxy.test:8080",
                "https_proxy": "http://codex-proxy.test:8080",
            })
            self.assertEqual(
                "http://codex-proxy.test:8080",
                runtime_api_config.get_codex_proxy_values()["http_proxy"],
            )

            runtime_api_config.apply_update({
                "target": "codex",
                "http_proxy": "",
                "https_proxy": "",
            })
            self.assertEqual(
                "http://base-proxy.test:8080",
                runtime_api_config.get_codex_proxy_values()["http_proxy"],
            )

    def test_runtime_config_refreshes_before_next_retry_attempt(self):
        with patch.dict(os.environ, {}, clear=True):
            connector = RetryRefreshConnector(self.tmpdir)
            response, _thinking, token_info = connector.send_message("hello", current_tick=1)

            self.assertEqual("ok", response)
            self.assertEqual(1, token_info["total_tokens_in_session"])
            self.assertEqual([0, 1], connector.attempt_generations)
            self.assertEqual(1, connector.refresh_count)

    def test_context_overflow_is_retried_before_success(self):
        connector = ContextOverflowRetryConnector(self.tmpdir, fail_count=2)

        response, _thinking, token_info = connector.send_message("hello", current_tick=1)

        self.assertEqual("ok", response)
        self.assertEqual(1, token_info["total_tokens_in_session"])
        self.assertEqual([0, 1, 2], connector.attempt_numbers)

    def test_context_overflow_raises_after_configured_attempt_cap(self):
        connector = ContextOverflowRetryConnector(self.tmpdir, fail_count=10, max_retries=0)

        with self.assertRaises(LLMContextOverflowError):
            connector.send_message("hello", current_tick=1)

        self.assertEqual(list(range(10)), connector.attempt_numbers)

    def test_provider_fallback_retries_same_endpoint_once_before_switching(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            connector = ProviderFallbackConnector(self.tmpdir)
            response, _thinking, token_info = connector.send_message("hello", current_tick=1)

            self.assertEqual("ok", response)
            self.assertEqual(1, token_info["total_tokens_in_session"])
            self.assertEqual([0, 0], connector.endpoint_indices)
            self.assertEqual([], connector.applied_indices)

    def test_provider_fallback_switches_endpoint_after_two_failures(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            connector = ProviderFallbackConnector(
                self.tmpdir,
                failures_before_success=2,
                max_retries=3,
            )
            response, _thinking, token_info = connector.send_message("hello", current_tick=1)

            self.assertEqual("ok", response)
            self.assertEqual(1, token_info["total_tokens_in_session"])
            self.assertEqual([0, 0, 1], connector.endpoint_indices)
            self.assertEqual([1], connector.applied_indices)

    def test_corrupted_thought_signature_bypasses_provider_fallback_and_retry(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            connector = ProviderFallbackConnector(
                self.tmpdir,
                error_cls=LLMCorruptedThoughtSignatureError,
                max_retries=3,
            )

            with self.assertRaises(LLMCorruptedThoughtSignatureError):
                connector.send_message("hello", current_tick=1)

            self.assertEqual([0], connector.endpoint_indices)
            self.assertEqual([], connector.applied_indices)

    def test_provider_fallback_waits_only_when_endpoint_loop_wraps(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            connector = ProviderFallbackConnector(
                self.tmpdir,
                failures_before_success=4,
                max_retries=5,
                retry_delay_seconds=7,
            )

            with patch("station.llm_connectors.base.time.sleep") as sleep_mock:
                response, _thinking, token_info = connector.send_message("hello", current_tick=1)

            self.assertEqual("ok", response)
            self.assertEqual(1, token_info["total_tokens_in_session"])
            self.assertEqual([0, 0, 1, 1, 0], connector.endpoint_indices)
            self.assertEqual([1, 0], connector.applied_indices)
            sleep_mock.assert_called_once_with(7)

    def test_context_overflow_can_fallback_to_next_provider_endpoint(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            connector = ProviderFallbackConnector(self.tmpdir, error_cls=LLMContextOverflowError)
            response, _thinking, token_info = connector.send_message("hello", current_tick=1)

            self.assertEqual("ok", response)
            self.assertEqual(1, token_info["total_tokens_in_session"])
            self.assertEqual([0, 0], connector.endpoint_indices)

    def test_provider_fallback_log_omits_generic_transient_message(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            connector = ProviderFallbackConnector(self.tmpdir)
            with patch("builtins.print") as print_mock:
                response, _thinking, token_info = connector.send_message("hello", current_tick=1)

            self.assertEqual("ok", response)
            self.assertEqual(1, token_info["total_tokens_in_session"])
            log_text = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
            self.assertNotIn("Retrying after transient error handling", log_text)
            self.assertIn("LLMConnector INFO (ProviderFallbackAgent): openai endpoint index=0 failed", log_text)
            self.assertIn("LLMTransientAPIError: first endpoint failed", log_text)

    def test_history_pruning_is_quiet_without_debug_api(self):
        connector = ContextOverflowRetryConnector(self.tmpdir, fail_count=0)

        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.print") as print_mock:
                result = connector._filter_and_prune_history([
                    {"tick": 1, "role": "user", "text_content": "hello"},
                ])

        self.assertEqual([{"role": "user", "text_content": "hello"}], result)
        log_text = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertNotIn("Before pruning", log_text)

    def test_pruned_summary_is_user_side_with_model_placeholder(self):
        connector = ContextOverflowRetryConnector(self.tmpdir, fail_count=0)
        connector.agent_prune_blocks = [
            {
                constants.PRUNE_TICKS_KEY: "123-145",
                constants.PRUNE_SUMMARY_KEY: "Range summary.",
            },
            {
                constants.PRUNE_TICKS_KEY: "151",
                constants.PRUNE_SUMMARY_KEY: "Single tick summary.",
            },
        ]

        result = connector._filter_and_prune_history([
            {"tick": 122, "role": "user", "text_content": "before"},
            {"tick": 123, "role": "user", "text_content": "range station"},
            {"tick": 145, "role": "model", "text_content": "range agent"},
            {"tick": 146, "role": "user", "text_content": "between"},
            {"tick": 151, "role": "model", "text_content": "single agent"},
            {"tick": 152, "role": "user", "text_content": "after"},
        ])

        self.assertEqual(
            [
                {"role": "user", "text_content": "before"},
                {
                    "role": "user",
                    "text_content": (
                        "Ticks 123-145 were pruned by the agent in the Token Management Room.\n"
                        "Summary submitted by the agent:\n"
                        "Range summary."
                    ),
                },
                {"role": "model", "text_content": "Dialogue pruned."},
                {"role": "user", "text_content": "between"},
                {
                    "role": "user",
                    "text_content": (
                        "Tick 151 was pruned by the agent in the Token Management Room.\n"
                        "Summary submitted by the agent:\n"
                        "Single tick summary."
                    ),
                },
                {"role": "model", "text_content": "Dialogue pruned."},
                {"role": "user", "text_content": "after"},
            ],
            result,
        )

    def test_final_retry_log_keeps_full_raw_exception(self):
        long_raw_message = "raw-provider-detail-" + ("x" * 1200)
        raw_error = RuntimeError(long_raw_message)
        connector_error = LLMTransientAPIError("wrapped failure", original_exception=raw_error)
        connector = ContextOverflowRetryConnector(self.tmpdir, fail_count=0)

        formatted = connector._format_exception_for_log(
            connector_error,
            include_raw=True,
            max_message_length=None,
        )

        self.assertIn(long_raw_message, formatted)
        self.assertNotIn("[truncated]", formatted)

    def test_gemini_client_creation_hides_duplicate_sdk_env_warning_source(self):
        connector = object.__new__(GoogleGeminiConnector)

        def fake_client(api_key, http_options=None):
            self.assertEqual("google-key", api_key)
            self.assertNotIn("GEMINI_API_KEY", os.environ)
            return SimpleNamespace(client_created=True)

        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "google-key",
            "GEMINI_API_KEY": "gemini-key",
        }, clear=True):
            with patch("station.llm_connectors.gemini.genai.Client", side_effect=fake_client):
                client = connector._create_genai_client("google-key", None)

            self.assertTrue(client.client_created)
            self.assertEqual("gemini-key", os.environ.get("GEMINI_API_KEY"))

    def test_context_overflow_provider_fallback_has_hard_attempt_cap(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            connector = ProviderFallbackConnector(
                self.tmpdir,
                error_cls=LLMContextOverflowError,
                always_fail=True,
            )

            with self.assertRaises(LLMContextOverflowError):
                connector.send_message("hello", current_tick=1)

            self.assertEqual(10, len(connector.endpoint_indices))
            self.assertEqual([0, 0, 1, 1, 0, 0, 1, 1, 0, 0], connector.endpoint_indices)

    def test_base_recovery_probe_uses_connector_model_and_restores_base_default(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            with patch("station.runtime_api_config.time.time", return_value=1000.0):
                for _ in range(10):
                    runtime_api_config.record_provider_failure_and_get_retry_snapshot(
                        "openai",
                        0,
                        ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                    )
            self.assertEqual(1, runtime_api_config.get_provider_default_endpoint("openai")["index"])

            connector = ProviderFallbackConnector(
                self.tmpdir,
                fail_first=False,
                probe_success=True,
            )
            with patch("station.runtime_api_config.time.time", return_value=2801.0):
                response, _thinking, _token_info = connector.send_message("hello", current_tick=1)

            self.assertEqual("ok", response)
            self.assertEqual([("test-model", 0)], connector.probe_calls)
            self.assertEqual([0], connector.endpoint_indices)
            self.assertEqual(0, runtime_api_config.get_provider_default_endpoint("openai")["index"])

    def test_base_recovery_probe_waits_thirty_minutes_after_promotion(self):
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "base-key",
            "BACKUP_OPENAI_API_KEY": "backup-key",
            "BACKUP_OPENAI_BASE_URL": "https://backup.example.test/v1",
        }, clear=True):
            with patch("station.runtime_api_config.time.time", return_value=1000.0):
                for _ in range(10):
                    runtime_api_config.record_provider_failure_and_get_retry_snapshot(
                        "openai",
                        0,
                        ["OPENAI_API_KEY", "OPENAI_BASE_URL"],
                    )

            connector = ProviderFallbackConnector(
                self.tmpdir,
                fail_first=False,
                probe_success=True,
            )
            with patch("station.runtime_api_config.time.time", return_value=2000.0):
                response, _thinking, _token_info = connector.send_message("hello", current_tick=1)

            self.assertEqual("ok", response)
            self.assertEqual([], connector.probe_calls)
            self.assertEqual([1], connector.endpoint_indices)
            self.assertEqual(1, runtime_api_config.get_provider_default_endpoint("openai")["index"])

    def test_openai_internal_stream_fallback_forces_non_stream_once(self):
        connector = make_openai_connector_for_internal_fallback_tests()
        token_info = {"total_tokens_in_session": None}

        with self.assertRaises(LLMConnectorError):
            connector._send_message_with_responses_api_stream(
                "hello",
                current_tick=1,
                token_info=token_info,
                attempt_number=0,
            )

        self.assertEqual(
            [True, None],
            [call.get("stream") for call in connector.client.responses.calls],
        )
        self.assertEqual(1, connector._refresh_count)

    def test_openai_internal_stream_503_defers_to_outer_retry(self):
        connector = make_openai_connector_for_internal_fallback_tests(
            FakeOpenAIStatusError(503, "No available channel for model gpt-5.5")
        )
        token_info = {"total_tokens_in_session": None}

        with self.assertRaises(LLMTransientAPIError):
            connector._send_message_with_responses_api_stream(
                "hello",
                current_tick=1,
                token_info=token_info,
                attempt_number=0,
            )

        self.assertEqual([True], [call.get("stream") for call in connector.client.responses.calls])
        self.assertEqual(0, connector._refresh_count)

    def test_openai_streaming_decision_uses_provider_proxy_snapshot(self):
        connector = make_openai_connector_for_internal_fallback_tests()
        constants.LLM_HTTP_PROXY = None
        constants.LLM_HTTPS_PROXY = None

        self.assertTrue(connector._should_use_streaming_on_attempt(0))

        connector._api_runtime_config_snapshot = {
            "generation": runtime_api_config.get_generation(),
            "http_proxy": None,
            "https_proxy": None,
            "env": {},
        }
        self.assertFalse(connector._should_use_streaming_on_attempt(0))


if __name__ == "__main__":
    unittest.main()
