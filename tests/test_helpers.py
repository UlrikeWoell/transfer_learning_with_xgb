from time import sleep

from src.helpers.mytimer import Timer


def test_timer():
    "test if time gets recorded with sufficient accuracy"
    timer = Timer()
    timer.start()
    sleep(2)
    timer.end()

    assert timer.duration > 1 and timer.duration < 3
