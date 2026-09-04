"""A benchmark for target-search route planning, with no simulator involved.

Answers one question: given a building, a belief about where an object is, and
the cost of walking between rooms, how much does an optimal visiting order
actually save over the obvious alternatives -- and what does computing it cost?

No Gazebo, no flight, no ROS. Buildings are generated procedurally as corridor
graphs, beliefs come from a parameterised model of a language-model oracle, and
every score is an exact expectation rather than a sampled trial. A full sweep
runs in minutes and is reproducible from its seeds.

See ``README.md``.
"""
