import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from station import constants
from station.multistart import branch_worker, state
from station.station_runner import Orchestrator


class FakeAgentModule:
    def __init__(self):
        self.data = {}

    def get_all_active_agent_names(self):
        return list(self.data)

    def load_agent_data(self, name):
        return self.data.get(name)

    def save_agent_data(self, name, data):
        self.data[name] = dict(data)
        return True


class FakeStation:
    def __init__(self):
        self.is_new_station = False
        self.config = {
            constants.STATION_CONFIG_CURRENT_TICK: 0,
            constants.STATION_CONFIG_AGENT_TURN_ORDER: [],
        }
        self.agent_module = FakeAgentModule()
        self.orchestrator = None

    def create_agent(self, **kwargs):
        name = f"Guest_{len(self.agent_module.data) + 1}"
        data = {
            constants.AGENT_NAME_KEY: name,
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_GUEST,
            constants.AGENT_SESSION_ENDED_KEY: False,
            constants.AGENT_IS_ASCENDED_KEY: False,
        }
        self.agent_module.data[name] = data
        return data, None

    def _save_config(self):
        return None

    def get_agent_departure_reason(self, _agent_name):
        return "unknown"

    def get_next_agent_index_from_config(self):
        return 0


class MultistartBranchInitTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_multistart_branch_init_test_", dir="/tmp")
        self.old_base = constants.BASE_STATION_DATA_PATH
        self.old_env = os.environ.get("STATION_MULTISTART_BRANCH")
        self.old_base_env = os.environ.get("STATION_BASE_DATA_PATH")
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        Path(self.tmpdir, constants.INIT_AGENTS_FILENAME).write_text("- Test Model\n", encoding="utf-8")

    def tearDown(self):
        constants.BASE_STATION_DATA_PATH = self.old_base
        if self.old_env is None:
            os.environ.pop("STATION_MULTISTART_BRANCH", None)
        else:
            os.environ["STATION_MULTISTART_BRANCH"] = self.old_env
        if self.old_base_env is None:
            os.environ.pop("STATION_BASE_DATA_PATH", None)
        else:
            os.environ["STATION_BASE_DATA_PATH"] = self.old_base_env
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @mock.patch("station.station_runner.runtime_api_config.validate_provider_backup_env_config")
    @mock.patch("station.station_runner.build_model_preset_lookup")
    def test_normal_init_spawns_agents_for_precreated_tick_zero_config(
        self,
        preset_lookup,
        _validate_runtime_config,
    ):
        os.environ.pop("STATION_MULTISTART_BRANCH", None)
        preset_lookup.return_value = {
            "Test Model": {
                "model_provider_class": "OpenAI",
                "model_name": "test-model",
            }
        }
        station = FakeStation()
        station.is_new_station = False
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.station = station
        orchestrator.is_prepared = True
        orchestrator.agent_turn_order = []
        orchestrator.agent_llm_connectors = {}
        orchestrator.is_running = False
        orchestrator.current_agent_index_in_turn_order = 0
        orchestrator.current_tick_processed_agents = set()
        orchestrator.launched = False
        orchestrator._push_log_event = lambda *_args, **_kwargs: None
        orchestrator.initialize_connector_for_agent = lambda agent_name, force_reinitialize=False: (
            orchestrator.agent_llm_connectors.setdefault(agent_name, object()) or True
        )
        orchestrator.start_processing_loop = lambda: setattr(orchestrator, "launched", True)

        orchestrator._try_init_agents_and_launch()

        self.assertEqual(["Guest_1"], station.config[constants.STATION_CONFIG_AGENT_TURN_ORDER])
        self.assertEqual(["Guest_1"], orchestrator.agent_turn_order)
        self.assertIn("Guest_1", orchestrator.agent_llm_connectors)
        self.assertTrue(orchestrator.launched)

    @mock.patch("station.station_runner.runtime_api_config.validate_provider_backup_env_config")
    @mock.patch("station.station_runner.build_model_preset_lookup")
    def test_multistart_branch_constructor_path_does_not_spawn_init_agents(
        self,
        preset_lookup,
        _validate_runtime_config,
    ):
        os.environ["STATION_MULTISTART_BRANCH"] = "1"
        preset_lookup.return_value = {
            "Test Model": {
                "model_provider_class": "OpenAI",
                "model_name": "test-model",
            }
        }
        station = FakeStation()
        station.is_new_station = False
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.station = station
        orchestrator.is_prepared = True
        orchestrator.agent_turn_order = []
        orchestrator.agent_llm_connectors = {}
        orchestrator.is_running = False
        orchestrator.current_agent_index_in_turn_order = 0
        orchestrator.current_tick_processed_agents = set()
        orchestrator._push_log_event = lambda *_args, **_kwargs: None
        orchestrator.initialize_connector_for_agent = lambda agent_name, force_reinitialize=False: (
            orchestrator.agent_llm_connectors.setdefault(agent_name, object()) or True
        )
        orchestrator.start_processing_loop = mock.Mock(side_effect=AssertionError("branch constructor path should not launch"))

        orchestrator._try_init_agents_and_launch()

        self.assertEqual([], station.config[constants.STATION_CONFIG_AGENT_TURN_ORDER])
        self.assertEqual([], orchestrator.agent_turn_order)
        self.assertEqual({}, orchestrator.agent_llm_connectors)

    @mock.patch("station.station_runner.runtime_api_config.validate_provider_backup_env_config")
    @mock.patch("station.station_runner.build_model_preset_lookup")
    def test_init_multistart_branch_spawns_init_agents_for_existing_tick_zero_config(
        self,
        preset_lookup,
        _validate_runtime_config,
    ):
        os.environ["STATION_MULTISTART_BRANCH"] = "1"
        preset_lookup.return_value = {
            "Test Model": {
                "model_provider_class": "OpenAI",
                "model_name": "test-model",
            }
        }
        station = FakeStation()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.station = station
        orchestrator.is_prepared = True
        orchestrator.agent_turn_order = []
        orchestrator.agent_llm_connectors = {}
        orchestrator.is_running = False
        orchestrator.current_agent_index_in_turn_order = 0
        orchestrator.current_tick_processed_agents = set()
        orchestrator._push_log_event = lambda *_args, **_kwargs: None
        orchestrator.initialize_connector_for_agent = lambda agent_name, force_reinitialize=False: (
            orchestrator.agent_llm_connectors.setdefault(agent_name, object()) or True
        )
        orchestrator.initialize_connectors_for_active_agents = lambda: True

        spawned = orchestrator.try_init_agents_for_multistart_branch()

        self.assertEqual(1, spawned)
        self.assertEqual(["Guest_1"], station.config[constants.STATION_CONFIG_AGENT_TURN_ORDER])
        self.assertEqual(["Guest_1"], orchestrator.agent_turn_order)
        self.assertIn("Guest_1", orchestrator.agent_llm_connectors)

    @mock.patch("station.multistart.interviews.run_interviews", return_value=True)
    @mock.patch("station.station.Station")
    @mock.patch("station.station_runner.Orchestrator")
    def test_stagnation_branch_does_not_spawn_init_agents(self, orchestrator_cls, station_cls, _run_interviews):
        tick = {"value": 0}
        station = SimpleNamespace(
            check_stagnation=lambda: None,
            _get_current_tick=lambda: tick["value"],
            config={},
        )
        station_cls.return_value = station
        orchestrator = SimpleNamespace(
            agent_llm_connectors={"Agent": object()},
            agent_turn_order=["Agent"],
            is_running=False,
            is_paused=False,
            try_init_agents_for_multistart_branch=mock.Mock(side_effect=AssertionError("init spawn called")),
            run_single_tick=lambda: tick.update(value=1) or True,
            stop_orchestration=lambda: None,
        )
        orchestrator_cls.return_value = orchestrator

        with tempfile.TemporaryDirectory(prefix="station_multistart_branch_job_test_", dir="/tmp") as job_dir:
            args = SimpleNamespace(
                data_root=self.tmpdir,
                job_dir=job_dir,
                seed=1,
                mode="stagnation",
                roll_ticks=1,
                branch_tick=0,
                poll_seconds=0.01,
            )
            branch_worker.run_branch(args)

        orchestrator.try_init_agents_for_multistart_branch.assert_not_called()

    @mock.patch("station.multistart.interviews.run_interviews", return_value=True)
    @mock.patch("station.station.Station")
    @mock.patch("station.station_runner.Orchestrator")
    def test_restarted_branch_without_target_uses_original_job_span(
        self,
        orchestrator_cls,
        station_cls,
        run_interviews,
    ):
        tick = {"value": 29}
        station = SimpleNamespace(
            check_stagnation=lambda: None,
            _get_current_tick=lambda: tick["value"],
            config={},
            has_pending_research_evaluations=lambda: False,
            has_pending_coder_sessions=lambda: False,
            has_pending_external_reports=lambda: False,
            has_pending_archive_surveys=lambda: False,
        )
        station_cls.return_value = station

        def run_single_tick():
            tick["value"] += 1
            return True

        orchestrator = SimpleNamespace(
            agent_llm_connectors={"Agent": object()},
            agent_turn_order=["Agent"],
            is_running=False,
            is_paused=False,
            try_init_agents_for_multistart_branch=mock.Mock(side_effect=AssertionError("init spawn called")),
            run_single_tick=mock.Mock(side_effect=run_single_tick),
            stop_orchestration=lambda: None,
        )
        orchestrator_cls.return_value = orchestrator

        with tempfile.TemporaryDirectory(prefix="station_multistart_branch_restart_test_", dir="/tmp") as job_dir:
            job_path = Path(job_dir)
            state.save_job_state(
                job_path,
                {
                    "branches": [
                        {
                            "seed": 1,
                            "status": "pending",
                            "current_tick": 29,
                        }
                    ],
                },
            )
            args = SimpleNamespace(
                data_root=self.tmpdir,
                job_dir=job_dir,
                seed=1,
                mode="stagnation",
                roll_ticks=40,
                branch_tick=0,
                poll_seconds=0.01,
            )
            rc = branch_worker.run_branch(args)
            payload = state.load_job_state(job_path)

        self.assertEqual(0, rc)
        branch = payload["branches"][0]
        self.assertEqual("completed", branch["status"])
        self.assertEqual(40, branch["target_tick"])
        self.assertEqual(40, branch["current_tick"])
        self.assertEqual(11, orchestrator.run_single_tick.call_count)
        run_interviews.assert_called_once()
        self.assertEqual(39, run_interviews.call_args.kwargs["base_tick"])

    def test_lazy_connectors_do_not_make_branch_unrunnable(self):
        with tempfile.TemporaryDirectory(prefix="station_multistart_lazy_connector_test_", dir="/tmp") as job_dir:
            job_path = Path(job_dir)
            state.save_job_state(
                job_path,
                {
                    "branches": [
                        {
                            "seed": 1,
                            "status": "running",
                        }
                    ],
                },
            )
            orchestrator = SimpleNamespace(
                agent_llm_connectors={},
                agent_turn_order=["Agent"],
            )

            branch_worker._ensure_runnable_branch(orchestrator, job_path, 1)
            payload = state.load_job_state(job_path)

        branch = payload["branches"][0]
        self.assertNotIn("reset_data_on_resume", branch)

    @mock.patch("station.multistart.interviews.run_interviews", side_effect=AssertionError("shutdown branch should not interview"))
    @mock.patch("station.station.Station")
    @mock.patch("station.station_runner.Orchestrator")
    def test_shutdown_requested_branch_drains_and_exits_without_new_tick(
        self,
        orchestrator_cls,
        station_cls,
        _run_interviews,
    ):
        tick = {"value": 5}
        station = SimpleNamespace(
            check_stagnation=lambda: None,
            _get_current_tick=lambda: tick["value"],
            config={},
            has_pending_research_evaluations=lambda: False,
            has_pending_coder_sessions=lambda: False,
            has_pending_external_reports=lambda: False,
            has_pending_archive_surveys=lambda: False,
        )
        station_cls.return_value = station
        orchestrator = SimpleNamespace(
            agent_llm_connectors={"Agent": object()},
            agent_turn_order=["Agent"],
            is_running=False,
            is_paused=False,
            try_init_agents_for_multistart_branch=mock.Mock(side_effect=AssertionError("init spawn called")),
            run_single_tick=mock.Mock(side_effect=AssertionError("shutdown branch should not start a new tick")),
            stop_orchestration=lambda: None,
        )
        orchestrator_cls.return_value = orchestrator

        with tempfile.TemporaryDirectory(prefix="station_multistart_branch_shutdown_test_", dir="/tmp") as job_dir:
            job_path = Path(job_dir)
            state.save_job_state(
                job_path,
                {
                    state.CONTROL_KEY: state.CONTROL_PAUSED,
                    state.SHUTDOWN_REQUESTED_KEY: True,
                    "branches": [{"seed": 1, "status": "running", "pid": 12345}],
                },
            )
            args = SimpleNamespace(
                data_root=self.tmpdir,
                job_dir=job_dir,
                seed=1,
                mode="stagnation",
                roll_ticks=10,
                branch_tick=0,
                poll_seconds=0.01,
            )
            rc = branch_worker.run_branch(args)
            payload = state.load_job_state(job_path)

        self.assertEqual(0, rc)
        branch = payload["branches"][0]
        self.assertEqual("paused", branch["status"])
        self.assertIsNone(branch["pid"])
        self.assertTrue(branch["shutdown_requested"])
        self.assertEqual("graceful multistart shutdown", branch["pause_reason"])
        orchestrator.run_single_tick.assert_not_called()


if __name__ == "__main__":
    unittest.main()
