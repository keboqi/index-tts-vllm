from __future__ import annotations

import unittest

from indextts_web.infrastructure.callables import filter_supported_keyword_arguments


class CallableCompatibilityTests(unittest.TestCase):
    def test_filters_options_missing_from_older_release(self) -> None:
        class OlderSeparator:
            def __init__(self, output_dir: str, use_autocast: bool = False) -> None:
                pass

        accepted, dropped = filter_supported_keyword_arguments(
            OlderSeparator,
            {
                "output_dir": "output",
                "use_soundfile": False,
                "use_autocast": True,
            },
        )

        self.assertEqual(
            accepted,
            {"output_dir": "output", "use_autocast": True},
        )
        self.assertEqual(dropped, ("use_soundfile",))

    def test_preserves_options_for_newer_release(self) -> None:
        class NewerSeparator:
            def __init__(
                self,
                output_dir: str,
                use_soundfile: bool = False,
                use_autocast: bool = False,
            ) -> None:
                pass

        options = {
            "output_dir": "output",
            "use_soundfile": True,
            "use_autocast": True,
        }
        accepted, dropped = filter_supported_keyword_arguments(
            NewerSeparator,
            options,
        )

        self.assertEqual(accepted, options)
        self.assertEqual(dropped, ())

    def test_preserves_options_for_kwargs_factory(self) -> None:
        def flexible_factory(**kwargs: object) -> object:
            return kwargs

        options = {"future_option": True}
        accepted, dropped = filter_supported_keyword_arguments(
            flexible_factory,
            options,
        )

        self.assertEqual(accepted, options)
        self.assertEqual(dropped, ())
