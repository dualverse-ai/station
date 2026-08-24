import unittest

from station.action_parser import ActionParser


class ActionParserRecoveryTests(unittest.TestCase):
    def test_recovers_glued_navigation_before_standalone_yaml_actions(self):
        response = """I am sending both messages now./execute_action{goto mail}

/execute_action{reply 602}
```yaml
title: Proposal decision
content: Proceed with a formal proposal.
```

/execute_action{create}
```yaml
recipients: Kairos IV
title: Regular meeting
content: Please send your report and plan.
```
The two messages have been sent.
"""

        actions = ActionParser().parse(response)

        self.assertEqual(
            [(action.command, action.args) for action in actions],
            [("goto", "mail"), ("reply", "602"), ("create", None)],
        )
        self.assertEqual(actions[1].yaml_data["title"], "Proposal decision")
        self.assertEqual(actions[2].yaml_data["recipients"], "Kairos IV")

    def test_leaves_ordinary_inline_example_inert(self):
        response = """I may use /execute_action{goto mail} in a later turn.
/execute_action{read 90}
"""

        actions = ActionParser().parse(response)

        self.assertEqual(
            [(action.command, action.args) for action in actions],
            [("read", "90")],
        )


if __name__ == "__main__":
    unittest.main()
