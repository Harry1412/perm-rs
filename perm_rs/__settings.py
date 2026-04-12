from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    _multi_threaded_threshold: int = 17

    @property
    def multi_threaded_threshold(self) -> int:
        """
        Sets the value of n for which the permanent computation switches from
        single to multi-threading.
        """
        return self._multi_threaded_threshold

    @multi_threaded_threshold.setter
    def multi_threaded_threshold(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                "Multi-threaded threshold must be set to an integer value."
            )
        if value < 0:
            raise ValueError("Multi-threaded threshold should not be negative.")
        self._multi_threaded_threshold = value


settings = Settings()
