import numpy as np

from src.preprocessing.compression import (
    generate_compression_levels
)


def compression_levels():

    return [95, 80, 60, 40, 20, 10]


def evaluate_compression(
    image,
    predictor
):

    results = {}

    versions = generate_compression_levels(
        image
    )

    for quality, compressed in versions.items():

        prediction = predictor(
            compressed
        )

        results[quality] = prediction

    return results
