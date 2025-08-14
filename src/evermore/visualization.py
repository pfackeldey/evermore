from __future__ import annotations

from collections.abc import Callable
from typing import Any

import treescope


class SupportsTreescope:
    def __treescope_repr__(
        self,
        path: str,
        subtree_renderer: Callable[
            [Any, str | None], treescope.rendering_parts.Rendering
        ],
    ) -> treescope.rendering_parts.Rendering:
        object_type = type(self)
        return treescope.repr_lib.render_object_constructor(
            object_type=object_type,
            attributes=dict(self.__dict__),
            path=path,
            subtree_renderer=subtree_renderer,
            # Pass `roundtrippable=True` only if you can rebuild your object by
            # calling `__init__` with these attributes!
            roundtrippable=True,
            color=treescope.formatting_util.color_from_string(object_type.__qualname__),
        )
