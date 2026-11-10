"""The Pandera schema every row of the dataset has to satisfy.

Kept apart from the stage that runs it so the schema can be unit tested in
CI without reading any images, and so the rules live in one readable place
rather than being scattered through a loop of if statements.
"""

import pandas as pd

# pandera 0.20 and newer want the pandas specific namespace; older versions
# only have the top level one, so try the new path first and fall back.
try:
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema

from nastaliqops.utils.urdu import is_mostly_urdu


MIN_TEXT_LENGTH = 2
MAX_TEXT_LENGTH = 300
MIN_IMAGE_WIDTH = 16
MIN_IMAGE_HEIGHT = 8
MAX_ASPECT_RATIO = 80.0
MAX_FAILURE_RATE = 0.05


def _check_urdu(series):
    """Element wise check, written as a plain loop for readability."""
    results = []
    for value in series:
        results.append(is_mostly_urdu(str(value)))
    return pd.Series(results, index=series.index)


MANIFEST_SCHEMA = DataFrameSchema(
    {
        "id": Column(int, nullable=False, unique=True),
        "file_name": Column(str, nullable=False, unique=True),
        "text": Column(
            str,
            checks=[
                Check.str_length(
                    min_value=MIN_TEXT_LENGTH,
                    max_value=MAX_TEXT_LENGTH,
                    name="text_length_in_range",
                ),
                Check(
                    _check_urdu,
                    name="text_is_mostly_urdu",
                    error="text_not_urdu",
                ),
            ],
            nullable=False,
        ),
        "width": Column(
            int,
            checks=[Check.greater_than_or_equal_to(MIN_IMAGE_WIDTH, name="image_wide_enough")],
            nullable=False,
        ),
        "height": Column(
            int,
            checks=[Check.greater_than_or_equal_to(MIN_IMAGE_HEIGHT, name="image_tall_enough")],
            nullable=False,
        ),
        "aspect_ratio": Column(
            float,
            checks=[Check.less_than_or_equal_to(MAX_ASPECT_RATIO, name="aspect_ratio_sane")],
            nullable=False,
        ),
    },
    strict=False,
    coerce=True,
)


def validate_frame(frame):
    """Run the schema and return (failed_ids, reason_counts).

    Pandera is asked for a lazy validation so it collects every failure
    instead of stopping at the first one, which is what makes the report
    useful rather than just a stack trace.
    """
    failed_ids = []
    reasons = {}

    try:
        MANIFEST_SCHEMA.validate(frame, lazy=True)
        return failed_ids, reasons
    except pa.errors.SchemaErrors as errors:
        cases = errors.failure_cases

        seen = {}
        for position in range(len(cases)):
            row = cases.iloc[position]

            check_name = row.get("check")
            if check_name is None:
                check_name = "unknown_check"
            check_name = str(check_name)

            if check_name in reasons:
                reasons[check_name] = reasons[check_name] + 1
            else:
                reasons[check_name] = 1

            index = row.get("index")
            if index is None:
                continue
            try:
                offending_id = int(frame.iloc[int(index)]["id"])
            except Exception:
                continue

            if offending_id not in seen:
                seen[offending_id] = True
                failed_ids.append(offending_id)

        return failed_ids, reasons
