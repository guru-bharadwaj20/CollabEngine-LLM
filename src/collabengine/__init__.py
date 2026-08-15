"""CollabEngine-LLM.

A framework for spinning up LLM agent teams with no assigned roles and causally
verifying whether the division of labor they develop is functionally real.

The central measurement is not a scalar performance drop when an agent is
ablated -- that only demonstrates participation. It is the agent x task-component
interaction: whether ablating an agent damages *its* components differentially.
See docs/PLAN.md.
"""

__version__ = "0.1.0"
