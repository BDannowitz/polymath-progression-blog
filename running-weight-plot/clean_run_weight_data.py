from math import floor
import pandas as pd

WG_DT_FORMAT = "%b %d %Y %I:%M:%S %p"
MFP_DT_FORMAT = "%Y-%m-%d"
RUNKEEPER_DT_FORMAT = "%Y-%m-%d %H:%M:%S"
STRAVA_DT_FORMAT = "%b %d, %Y, %I:%M:%S %p"

WEIGHT_COLS = ["Date", "Weight"]
RUN_COLS = ['Date', 'Name', 'Description', 'Distance', 'Pace',
            'Duration', 'Pace_min', 'Duration_min', 'Tracker']


def decimal_minute_to_time(dec_minutes):
    """Converts decimal minutes to MM:SS format."""
    hour = floor(dec_minutes / 60)
    minute = int(dec_minutes % 60)
    sec = int(60 * (dec_minutes - int(dec_minutes)))
    
    time_str = ""
    if hour > 0:
        time_str = "{}:{:02}:{:02}".format(hour, minute, sec)
    else:
        time_str = "{}:{:02}".format(minute, sec)
    return time_str


def time_to_decimal_minute(time_str):
    """Converts MM:SS or HH:MM:SS string to decimal minute format."""
    time_list = time_str.split(":")
    minute, second = int(time_list[-2]), int(time_list[-1])
    if len(time_list) == 3:
        minute = minute + 60.0 * int(time_list[0])
    if second >= 60:
        raise ValueError("Bad time string format. More than 60s: %s", second)
    dec_minute = minute + second/60.0 
    return dec_minute


def process_weight_gurus(wg_filename):
    weight_gurus = pd.read_csv(wg_filename)
    weight_gurus = weight_gurus.rename(columns={'Date/Time': 'Date'})
    weight_gurus['Date'] = weight_gurus['Date'].apply(
        lambda x: datetime.strptime(x, WG_DT_FORMAT)
    )
    weight_gurus = weight_gurus.rename(
        columns={'Weight (lb)': 'Weight'}
    )
    
    return weight_gurus

def process_mfp_weight(mfp_filename):
    mfp = pd.read_csv(mfp_filename)
    mfp = mfp.dropna(subset=['Weight'])
    mfp['Date'] = mfp['Date'].apply(
        lambda x: datetime.strptime(x, MFP_DT_FORMAT)
    )
    return mfp

def process_runkeeper(runkeeper_filename):
    runkeeper = pd.read_csv(runkeeper_filename)
    runkeeper = runkeeper[runkeeper['Type'] == "Running"]
    runkeeper['Date'] = runkeeper['Date'].apply(
        lambda x: datetime.strptime(x, RUNKEEPER_DT_FORMAT)
    )

    runkeeper = runkeeper.rename(columns={'Distance (mi)': 'Distance',
                                          'Notes': 'Name',
                                          'Average Pace': 'Pace'})

    runkeeper['Pace_min'] = runkeeper['Pace'].apply(time_to_decimal_minute)
    runkeeper['Duration_min'] = runkeeper['Duration'].apply(time_to_decimal_minute)

    runkeeper['Description'] = None
    runkeeper['Tracker'] = "Runkeeper"
    
    return runkeeper

def process_strava(strava_filename):
    strava = pd.read_csv(strava_filename)
    strava = strava[strava['Activity Type'] == "Run"]
    
    strava = strava.rename(columns={'Activity Date': 'Date',
                                    'Activity Name': 'Name',
                                    'Activity Description': 'Description'})
    strava['Date'] = strava['Date'].apply(
        lambda x: datetime.strptime(x, STRAVA_DT_FORMAT)
    )
    
    # Convert km -> mi
    strava['Distance'] = strava['Distance'] * 0.621371

    # Calculate pace (in decimal minutes)
    strava['Pace_min'] = strava['Elapsed Time'] / (60 * strava['Distance'])

    # Calculate duration (in decimal minutes)
    strava['Duration_min'] = strava['Elapsed Time']/60.0

    # Convert decimal minute to MM:SS
    strava['Pace'] = strava['Pace_min'].apply(decimal_minute_to_time)
    strava['Duration'] = strava['Duration_min'].apply(decimal_minute_to_time)

    strava['Tracker'] = 'Strava'
    
    return strava

def combine_weights(df_list, weight_cols=WEIGHT_COLS):
    weight_df = pd.concat([df[weight_cols] for df in df_list])
    weight_df = weight_df.sort_values('Date')
    return weight_df

def combine_runs(df_list, run_cols=RUN_COLS):
    run_df = pd.concat([df[run_cols] for df in df_list])
    run_df = run_df.sort_values('Date')
    return run_df

def main():
    strava = process_strava('strava-activities.csv')
    runkeeper = process_runkeeper('runkeeper-activities.csv')
    
    mfp = process_mfp_weight('myfitnesspal-export.csv')
    weight_gurus = process_weight_gurus('weight-gurus-history.csv')
    
    run_df = combine_runs([strava, runkeeper])
    weight_df = combine_weights([mfp, weight_gurus])
    
    run_df.to_csv('run.csv')
    weight_df.to_csv('weight.csv')

if __name__ == "__main__":
    print("processing data beep boop bonk")
    main()