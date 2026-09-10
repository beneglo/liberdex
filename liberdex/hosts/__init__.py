"""Host adapters: liberdex code that runs inside another agent's process.

`liberdex mcp` is a process of its own and speaks to every host the same way.
An adapter here is for a host that wants a web search *provider* in its own
process, with its own model writing the plan. Each one is loaded by the host
through an entry point the user enables; the host never imports liberdex.
"""
