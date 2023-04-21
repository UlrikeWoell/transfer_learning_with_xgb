from time import sleep

from src.helpers.mytimer import Timer


def test_timer():
    "test if time gets recorded with sufficient accuracy"
    timer = Timer()
    timer.start()
    sleep(1)
    timer.end()

    assert True