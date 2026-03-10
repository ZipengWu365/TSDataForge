TAXONOMY = {
    "statistical": [
        "statistical.white_noise",
        "statistical.colored_noise",
        "statistical.ar1",
        "statistical.random_walk",
    ],
    "trend": [
        "trend.linear",
        "trend.piecewise_linear",
    ],
    "periodic": [
        "periodic.single",
        "periodic.multi",
        "periodic.quasi",
    ],
    "dynamic": [
        "dynamic.regime_switching",
        "control.closed_loop",
        "control.second_order_tracking",
        "control.event_triggered",
    ],
    "events": [
        "events.bursty",
        "event_driven",
        "hybrid_system",
    ],
    "control": [
        "control.reference",
        "control.reference.step",
        "control.reference.waypoint",
        "control.closed_loop",
        "control.second_order_tracking",
        "control.event_triggered",
        "robotics",
    ],
    "composition": [
        "composition.add",
        "composition.multiply",
        "composition.convolve",
        "composition.time_warp",
    ],
    "observation": [
        "irregular_sampling",
        "missingness.block",
        "measurement_noise",
        "downsampled",
    ],
}
