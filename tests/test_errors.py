from bold_reliability_movies.errors import (
    BrmError,
    EmptyDiscoveryError,
    EncodeError,
    GroupRejectedError,
    InconsistentShapesError,
    MissingDependency,
    UnknownRendererError,
)


def test_all_inherit_from_brm_error():
    for cls in (
        MissingDependency,
        InconsistentShapesError,
        EncodeError,
        EmptyDiscoveryError,
        UnknownRendererError,
        GroupRejectedError,
    ):
        assert issubclass(cls, BrmError)


def test_brm_error_inherits_from_exception():
    assert issubclass(BrmError, Exception)


def test_inconsistent_shapes_error_carries_shapes():
    err = InconsistentShapesError(
        frame_index=4,
        previous_shape=(320, 320, 3),
        current_shape=(360, 360, 3),
        suggestion="use --group-by subject+task",
    )
    msg = str(err)
    assert "frame 4" in msg.lower()
    assert "320" in msg
    assert "360" in msg
    assert "--group-by subject+task" in msg


def test_unknown_renderer_lists_available():
    err = UnknownRendererError(name="flatmap", available=["mosaic", "triplet"])
    msg = str(err)
    assert "flatmap" in msg
    assert "mosaic" in msg
    assert "triplet" in msg
    assert err.name == "flatmap"
    assert err.available == ["mosaic", "triplet"]
