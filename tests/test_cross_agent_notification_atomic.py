import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List

from station import agent as agent_module
from station import constants
from station.base_room import RoomContext
from station.rooms.common import CommonRoom
from station.rooms.public_memory import PublicMemoryRoom


class _StationStub:
    def _get_current_tick(self) -> int:
        return 5

    def _should_agent_receive_broadcast(
        self,
        agent_data: Dict[str, Any],
        current_tick: int,
        broadcast_type: str = "general",
    ) -> bool:
        return True

    def _is_agent_mature(self, agent_data: Dict[str, Any], current_tick: int) -> bool:
        return True


class _ConcurrentNotificationAgentManager:
    add_pending_notification = staticmethod(agent_module.add_pending_notification)

    def __init__(self, active_names: List[str], concurrent_agent: str, concurrent_message: str):
        self._active_names = active_names
        self._concurrent_agent = concurrent_agent
        self._concurrent_message = concurrent_message
        self._injected = False

    def get_all_active_agent_names(self) -> List[str]:
        return list(self._active_names)

    def load_agent_data(self, agent_name: str):
        return agent_module.load_agent_data(agent_name)

    def save_agent_data(self, agent_name: str, agent_data: Dict[str, Any]) -> bool:
        if agent_name == self._concurrent_agent:
            self._inject_concurrent_notification()
        return agent_module.save_agent_data(agent_name, agent_data)

    def update_agent_with_function(self, agent_name: str, update_func) -> bool:
        if agent_name == self._concurrent_agent:
            self._inject_concurrent_notification()
        return agent_module.update_agent_with_function(agent_name, update_func)

    def _inject_concurrent_notification(self) -> None:
        if self._injected:
            return
        self._injected = True
        agent_module.add_pending_notification_atomic(
            self._concurrent_agent,
            self._concurrent_message,
        )


class CrossAgentNotificationAtomicTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="station_cross_agent_atomic_", dir="/tmp")
        self.old_base = constants.BASE_STATION_DATA_PATH
        constants.BASE_STATION_DATA_PATH = self.tmpdir
        os.makedirs(os.path.join(self.tmpdir, constants.AGENTS_DIR_NAME), exist_ok=True)

    def tearDown(self):
        constants.BASE_STATION_DATA_PATH = self.old_base
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _save_agent(self, name: str) -> None:
        agent_data = {
            constants.AGENT_NAME_KEY: name,
            constants.AGENT_STATUS_KEY: constants.AGENT_STATUS_RECURSIVE,
            constants.AGENT_SESSION_ENDED_KEY: False,
            constants.AGENT_IS_ASCENDED_KEY: False,
            constants.AGENT_NOTIFICATIONS_PENDING_KEY: [],
            constants.AGENT_SHOWN_NOTIFICATIONS_KEY: [],
        }
        self.assertTrue(agent_module.save_agent_data(name, agent_data))

    def _room_context(self, manager: _ConcurrentNotificationAgentManager) -> RoomContext:
        return RoomContext(
            agent_manager=manager,
            capsule_manager=None,
            notification_manager=None,
            constants_module=constants,
            station_instance=_StationStub(),
        )

    def test_public_memory_broadcast_preserves_concurrent_notification(self):
        self._save_agent("Author")
        self._save_agent("Recipient")
        background_message = "background evaluator finished"
        manager = _ConcurrentNotificationAgentManager(
            ["Author", "Recipient"],
            "Recipient",
            background_message,
        )

        PublicMemoryRoom()._after_capsule_created(
            {
                constants.CAPSULE_AUTHOR_NAME_KEY: "Author",
                constants.CAPSULE_TITLE_KEY: "Atomic Thread",
                constants.CAPSULE_ID_KEY: "public_1",
                constants.CAPSULE_WORD_COUNT_TOTAL_KEY: 12,
                constants.CAPSULE_MESSAGES_KEY: [
                    {constants.MESSAGE_CONTENT_KEY: "Opening note without mentions."}
                ],
            },
            {},
            self._room_context(manager),
            current_tick=5,
        )

        updated = agent_module.load_agent_data("Recipient")
        notifications = updated[constants.AGENT_NOTIFICATIONS_PENDING_KEY]
        self.assertIn(background_message, notifications)
        self.assertTrue(
            any("A new public memory capsule (#1)" in message for message in notifications)
        )

    def test_common_invite_preserves_concurrent_notification(self):
        self._save_agent("Sender")
        self._save_agent("Recipient")
        background_message = "background evaluator finished"
        manager = _ConcurrentNotificationAgentManager(
            ["Sender", "Recipient"],
            "Recipient",
            background_message,
        )
        sender_data = agent_module.load_agent_data("Sender")

        actions, _ = CommonRoom().handle_action(
            sender_data,
            constants.ACTION_COMMON_INVITE,
            None,
            {constants.YAML_COMMON_RECIPIENTS: ["Recipient"]},
            self._room_context(manager),
            current_tick=5,
        )

        updated = agent_module.load_agent_data("Recipient")
        notifications = updated[constants.AGENT_NOTIFICATIONS_PENDING_KEY]
        self.assertIn("Sent 1 invitation(s) to the Common Room.", actions)
        self.assertIn(background_message, notifications)
        self.assertTrue(
            any("invites you to join them in the Common Room" in message for message in notifications)
        )


if __name__ == "__main__":
    unittest.main()
