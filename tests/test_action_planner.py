from app.planner.action_planner import plan_action


def test_plan_action_transfer():
    action = plan_action(
        intent="TRANSFER",
        entities={
            "accountNumber": "123456789",
            "bank": "MB BANK",
            "amount": 2000000,
            "description": "Thanh toan hoa don",
        },
    )
    assert action is not None
    assert action["screen"] == "TRANSFER_SCREEN"
    assert action["fields"]["account"] == "123456789"


def test_plan_action_unknown():
    assert plan_action(intent="UNKNOWN", entities={}) is None
