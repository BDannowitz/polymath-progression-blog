import pandas as pd
import numpy as np

import time
from datetime import datetime as dt
from bokeh.models import (
    LinearAxis, Range1d, ColumnDataSource,
    LogColorMapper, HoverTool,
    Arrow, NormalHead, LabelSet
)
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Plasma10

Plasma10.reverse()


def get_run_df(run_filepath):
    
    run_df = (pd.read_csv(run_filepath, parse_dates=['Date'], index_col=0)
          .reset_index(drop=True))
    run_df = run_df.fillna('')
    
    # Highlighting the races on the visualization
    run_df['Line_Color'] = "#C0C0C0"  # Silver
    run_df.loc[[237, 127], 'Line_Color'] = "#39FF14"  # Neon Green
    
    # I'm tying distance to size, but need the numbers to be a little larger...
    run_df['Display_Size'] = run_df['Distance'] * 2
    
    return run_df

    
def get_weight_df(weight_filepath):
    weight_df = pd.read_csv(weight_filepath, parse_dates=['Date'], index_col=0)
    weight_df = weight_df[weight_df['Date'] > '2017-02-28']
    weight_df = weight_df.reset_index(drop=True)
    
    return weight_df


def poly_fit_weight(weight_df):
    
    # Turn datetime into unix time (for fitting purposes)
    weight_df['timestamp'] = weight_df.Date.apply(
        lambda x: int(round(x.timestamp()))
    )

    # Fit the polynomial. 3 degrees should do just fine, given the shape
    params = np.polyfit(weight_df['timestamp'], weight_df['Weight'], 3)

    # Get an effective x-range for the fit to display on
    date_range = pd.date_range(start=weight_df.Date.min(),
                               end=weight_df.Date.max(),
                               freq='D', )
    date_range_sec = [x.timestamp() for x in date_range]

    # Get polynomial function value for each day in that range
    poly_func = np.poly1d(params)
    poly_vals = poly_func(date_range_sec)

    # Toss into a dataframe for plotting
    poly_df = pd.DataFrame({'Date': date_range, 'Weight': poly_vals})
    
    return poly_df


def make_viz(run_df, weight_df, poly_df, output_filename):
    
    # Where are we shipping this out to?
    output_file(output_filename)

    # Create a new plot with a datetime axis type
    p = figure(plot_width=1100,
               plot_height=500,
               x_axis_type="datetime",
               title="Run Log and Weigh-Ins",
               toolbar_location="above")

    # Create a second y-axis, set y-axis ranges
    p.extra_y_ranges = {"pace": Range1d(start=12, end=7.33)}
    p.y_range = Range1d(170, 245)

    # Adding the second axis to the plot.  
    p.add_layout(LinearAxis(y_range_name="pace"), 'right')

    # Label the axes
    p.xaxis.axis_label = "Date"
    p.yaxis[0].axis_label = "Weight (lbs)"
    p.yaxis[1].axis_label = "Pace (Minutes per Mile)"

    # Redundant, but it looks cool.
    log_cmap = LogColorMapper(palette=Plasma10, low=7.5, high=11.5)

    # Plot running logs
    run_glyph = p.circle(
        x='Date',
        y='Pace_min',
        size='Display_Size',
        line_color='Line_Color',
        line_width=2,
        fill_color={'field': 'Pace_min', 'transform': log_cmap},
        alpha=0.8,
        y_range_name="pace",
        source=run_df,
        legend="Run Log"
    )

    # Add hover-over details for run log
    tooltips = [
        ("Date", "@Date{%F}"),
        ("Distance", "@Distance{1.1} mi"),
        ("Pace", "@Pace min/mi"),
        ("Duration", "@Duration"),
        ("Name", "@Name"),
        ("Description", "@Description"),
        ("Tracker", "@Tracker"),
    ]
    p.add_tools(
        HoverTool(
            tooltips=tooltips,
            renderers=[run_glyph],
            formatters={"Date": "datetime"}
        )
    )

    # Plot weigh-ins as small dots
    p.circle(x='Date', y='Weight', size=2, color='gray',
             source=weight_df, alpha=0.7)

    # Plot the polynomial fit
    p.line(x='Date', y='Weight', source=poly_df, legend="Weight (Poly Fit)")

    # Legend tuning
    p.legend.location = 'bottom_left'
    p.legend.background_fill_alpha = 0.5

    ###### Start Annotations

    # Time Events
    new_house = time.mktime(dt(2018, 1, 15, 0, 0, 0).timetuple())*1000
    kc_half_train = time.mktime(dt(2018, 5, 1, 0, 0, 0).timetuple())*1000
    tendonitis_IF = time.mktime(dt(2019, 2, 1, 0, 0, 0).timetuple())*1000

    # Create arrows that point to spots along the timeline
    arrows_df = pd.DataFrame(
        {'x_start': [new_house, kc_half_train, tendonitis_IF],
         'x_end': [new_house, kc_half_train, tendonitis_IF],
         'y_start': [180, 220, 179],
         'y_end': [171, 171, 171]})
    p.add_layout(Arrow(end=NormalHead(size=10, fill_color="gray"),
                       line_width=3, line_color="#666666",
                       x_start='x_start', x_end='x_end',
                       y_start='y_start', y_end='y_end',
                       source=ColumnDataSource(arrows_df)))

    # Create labels for these moments
    labels_df = pd.DataFrame(
        {'x': [new_house, kc_half_train, tendonitis_IF, tendonitis_IF],
         'y': [182, 223, 184.5, 181],
         'text': ["New House", "Start KC Half Marathon Training",
                  "Achilles Tendonitis", "Started 16:8 IF"]}
    )
    p.add_layout(
        LabelSet(
            x="x", y="y", text="text",
            source=ColumnDataSource(labels_df),
            text_baseline="middle", text_align="center",
            text_color="#666666"
        )
    )

    # Ship it.
    show(p)


def main():
    
    run_df = get_run_df('data/processed/run.csv')
    weight_df = get_weight_df('data/processed/weight.csv')
    poly_df = poly_fit_weight(weight_df)
    
    make_viz(run_df, weight_df, poly_df, "html/bdannowitz-fitness-viz.html")


if __name__ == "__main__":
    main()
