"""Shared page controls, independent of audio/video rendering and models."""
import inspect

import streamlit as st


def saved_control(widget, name, *args, **kwargs):
    """Keep preferences outside widget state, which is cleared on page changes."""
    key = f"_control_{name}"
    arguments = inspect.signature(widget).bind_partial(*args, **kwargs)
    if name in st.session_state:
        if "options" in arguments.arguments:
            arguments.arguments["index"] = list(arguments.arguments["options"]).index(st.session_state[name])
        else:
            arguments.arguments["value"] = st.session_state[name]
    value = widget(*arguments.args, key=key, **arguments.kwargs)
    st.session_state[name] = value
    return value
