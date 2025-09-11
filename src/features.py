# src/features.py
import pandas as pd

# I'm defining the list of feature columns here as a global constant.
# This makes it easy to import and use consistently across my project,
# which helps prevent bugs from typos or forgetting a feature later on.
feature_cols = [
    # Direct comparison features
    "ReachDiff", "AgeDiff", "HeightDiff", "WeightDiff", "KODiff", "WinDiff",
    "RecentWinPercDiff", "DefenseDiff", "PunchAccDiff", "SoSDiff", "TimeSinceDiff",

    # My custom interaction features
    "PowerVsSchedule", "WinsVsSchedule", "RecentFormWeighted"
]


def engineer_features(df_in):
    """
    In this function, I create new, more powerful features from the raw fighter stats.

    I realised that the model would learn more from the difference or relative
    advantage between the two opponents, rather than just their raw stats. For example,
    a fighters height isn't as important as their height advantage over their opponent.

    Args:
        df_in (pd.DataFrame): The input DataFrame with raw stats for Fighter A and B.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with my new engineered feature columns.
            - list[str]: A list of the names of the feature columns I created.
    """
    # I'm working on a copy here to make sure I don't change the original DataFrame.
    df = df_in.copy()

    # My Primary Difference Features
    # These features directly capture the physical and statistical advantages of
    # Fighter A over Fighter B. A positive value means A has an advantage on that
    # metric, while a negative value means B has the advantage.
    df["ReachDiff"]         = df["ReachA"] - df["ReachB"]
    df["AgeDiff"]           = df["AgeA"] - df["AgeB"]
    df["HeightDiff"]        = df["HeightA"] - df["HeightB"]
    df["WeightDiff"]        = df["WeightA"] - df["WeightB"]
    df["KODiff"]            = df["KOPercA"] - df["KOPercB"]
    df["WinDiff"]           = df["WinsA"] - df["WinsB"]
    df["RecentWinPercDiff"] = df["RecentWinPercA"] - df["RecentWinPercB"]
    df["DefenseDiff"]       = df["DefensePercA"] - df["DefensePercB"]
    df["PunchAccDiff"]      = df["PunchAccuracyA"] - df["PunchAccuracyB"]
    df["SoSDiff"]           = df["StrengthOfScheduleA"] - df["StrengthOfScheduleB"]
    df["TimeSinceDiff"]     = df["TimeSinceLastFightA"] - df["TimeSinceLastFightB"]

    # My Custom Interaction Features
    # Here, I combined some of the basic differences to create more nuanced features
    # that I thought would be highly relevant for predicting a boxing match.

    # 'PowerVsSchedule': I created this to measure knockout power, adjusted for opponent quality.
    # I reasoned that a high KO percentage is more meaningful if it was achieved against tough opponents.
    df["PowerVsSchedule"]   = df["KODiff"] * df["SoSDiff"]

    # 'WinsVsSchedule': This is similar to the above. I wanted to adjust a fighter's win record
    # by the strength of their schedule.
    df["WinsVsSchedule"]    = df["WinDiff"] * df["SoSDiff"]

    # 'RecentFormWeighted': With this feature, I took the recent win percentage difference and
    # boosted it based on the strength of schedule. This rewards fighters who are not only
    # winning, but winning against strong opponents.
    sos_bonus = 0.5 * (df["SoSDiff"] / (1 + df["SoSDiff"].abs()))
    df["RecentFormWeighted"] = df["RecentWinPercDiff"] * (1 + sos_bonus)

    # This loop is a safeguard I added. It ensures the DataFrame returned by this function
    # always contains all the columns I've defined in `feature_cols`, preventing errors
    # in my downstream model training step.
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df, feature_cols