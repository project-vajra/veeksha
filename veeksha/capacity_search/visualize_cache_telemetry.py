#!/usr/bin/env python3
import json
import altair as alt
import pandas as pd
from datetime import datetime
import os
import glob
import argparse

def load_telemetry_data(file_path):
    """Load telemetry data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_block_level_charts(data):
    """Create block level charts from telemetry data."""
    # Extract block level data
    block_level = data['block_level']
    
    # Create a DataFrame for the pie chart
    pie_data = pd.DataFrame({
        'category': ['Hits', 'Misses'],
        'value': [block_level['hits'], block_level['misses']]
    })
    
    # Create pie chart for hit/miss ratio
    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="category", type="nominal", 
                       scale=alt.Scale(domain=['Hits', 'Misses'], 
                                      range=['#2ecc71', '#e74c3c'])),
        tooltip=['category', 'value']
    ).properties(
        title=f"Block Level Hit/Miss (Rate: {block_level['hit_rate']:.2%})",
        width=300,
        height=300
    )
    
    # Create time series data for block level
    ts_data = []
    
    # Process time series data
    cumulative_total = 0
    cumulative_hits = 0
    cumulative_misses = 0
    
    # Sort by timestamp to ensure correct cumulative calculation
    total_blocks_ts = sorted(data['block_level_ts'].get('total_blocks', []), key=lambda x: x[0])
    hits_ts = sorted(data['block_level_ts'].get('hits', []), key=lambda x: x[0])
    misses_ts = sorted(data['block_level_ts'].get('misses', []), key=lambda x: x[0])
    
    # Process total blocks (cumulative)
    for timestamp, value in total_blocks_ts:
        cumulative_total += value
        ts_data.append({
            'timestamp': timestamp,
            'value': cumulative_total,
            'series': 'Total Blocks'
        })
    
    # Process hits (cumulative)
    for timestamp, value in hits_ts:
        cumulative_hits += value
        ts_data.append({
            'timestamp': timestamp,
            'value': cumulative_hits,
            'series': 'Cumulative Hits'
        })
    
    # Process misses (cumulative)
    for timestamp, value in misses_ts:
        cumulative_misses += value
        ts_data.append({
            'timestamp': timestamp,
            'value': cumulative_misses,
            'series': 'Cumulative Misses'
        })
    
    if not ts_data:
        return pie_chart
    
    ts_df = pd.DataFrame(ts_data)
    
    # Create time series chart with cumulative values
    time_series = alt.Chart(ts_df).encode(
        x=alt.X('timestamp:Q', title='Time (seconds)'),
        y=alt.Y('value:Q', title='Blocks'),
        color=alt.Color('series:N', 
                       scale=alt.Scale(domain=['Total Blocks', 'Cumulative Hits', 'Cumulative Misses'], 
                                      range=['#3498db', '#2ecc71', '#e74c3c'])),
        tooltip=['timestamp', 'value', 'series'],
        strokeDash=alt.StrokeDash('series:N', 
                                 scale=alt.Scale(domain=['Total Blocks', 'Cumulative Hits', 'Cumulative Misses'],
                                               range=[[1, 0], [2, 2], [5, 2]])),
        strokeWidth=alt.StrokeWidth('series:N',
                                   scale=alt.Scale(domain=['Total Blocks', 'Cumulative Hits', 'Cumulative Misses'],
                                                 range=[3, 2, 2]))
    ).mark_line(point=True).properties(
        title="Block Level Activity Over Time",
        width=600,
        height=300
    )
    
    # Add a selector for choosing which series to display
    selection = alt.selection_point(fields=['series'], bind='legend')
    
    # Apply the selector to the chart
    time_series = time_series.add_params(
        selection
    ).encode(
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    )
    
    return pie_chart, time_series

def create_request_level_charts(data):
    """Create request level charts from telemetry data."""
    # Extract request level data
    request_level = data['request_level']
    
    # Create a DataFrame for the pie chart
    pie_data = pd.DataFrame({
        'category': ['Hits', 'Misses'],
        'value': [request_level['hits'], request_level['misses']]
    })
    
    # Create pie chart for hit/miss ratio
    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="category", type="nominal", 
                       scale=alt.Scale(domain=['Hits', 'Misses'], 
                                      range=['#2ecc71', '#e74c3c'])),
        tooltip=['category', 'value']
    ).properties(
        title=f"Request Level Hit/Miss (Rate: {request_level['hit_rate']:.2%})",
        width=300,
        height=300
    )
    
    # Create time series data for request level
    ts_data = []
    
    # Process time series data
    cumulative_requests = 0
    cumulative_hits = 0
    cumulative_misses = 0
    
    # Sort by timestamp to ensure correct cumulative calculation
    unique_requests_ts = sorted(data['request_level_ts'].get('unique_requests', []), key=lambda x: x[0])
    hits_ts = sorted(data['request_level_ts'].get('hits', []), key=lambda x: x[0])
    misses_ts = sorted(data['request_level_ts'].get('misses', []), key=lambda x: x[0])
    
    # Process unique requests (cumulative)
    for timestamp, value in unique_requests_ts:
        cumulative_requests += value
        ts_data.append({
            'timestamp': timestamp,
            'value': cumulative_requests,
            'series': 'Requests'
        })
    
    # Process hits (cumulative)
    for timestamp, value in hits_ts:
        cumulative_hits += value
        ts_data.append({
            'timestamp': timestamp,
            'value': cumulative_hits,
            'series': 'Cumulative Hits'
        })
    
    # Process misses (cumulative)
    for timestamp, value in misses_ts:
        cumulative_misses += value
        ts_data.append({
            'timestamp': timestamp,
            'value': cumulative_misses,
            'series': 'Cumulative Misses'
        })
    
    if not ts_data:
        return pie_chart
    
    ts_df = pd.DataFrame(ts_data)
    
    # Create time series chart
    time_series = alt.Chart(ts_df).encode(
        x=alt.X('timestamp:Q', title='Time (seconds)'),
        y=alt.Y('value:Q', title='Requests'),
        color=alt.Color('series:N', 
                       scale=alt.Scale(domain=['Requests', 'Cumulative Hits', 'Cumulative Misses'], 
                                      range=['#3498db', '#2ecc71', '#e74c3c'])),
        tooltip=['timestamp', 'value', 'series'],
        strokeDash=alt.StrokeDash('series:N', 
                                 scale=alt.Scale(domain=['Requests', 'Cumulative Hits', 'Cumulative Misses'],
                                               range=[[1, 0], [2, 2], [5, 2]])),
        strokeWidth=alt.StrokeWidth('series:N',
                                   scale=alt.Scale(domain=['Requests', 'Cumulative Hits', 'Cumulative Misses'],
                                                 range=[3, 2, 2]))
    ).mark_line(point=True).properties(
        title="Request Level Activity Over Time",
        width=600,
        height=300
    )
    
    # Add a selector for choosing which series to display
    selection = alt.selection_point(fields=['series'], bind='legend')
    
    # Apply the selector to the chart
    time_series = time_series.add_params(
        selection
    ).encode(
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    )
    
    return pie_chart, time_series

def visualize_telemetry(data, output_dir=None, filename=None):
    """Create and save visualizations for telemetry data."""
    # Create charts
    block_pie, block_time_series = create_block_level_charts(data)
    request_pie, request_time_series = create_request_level_charts(data)
    
    # Create combined pie chart
    combined_pie = alt.hconcat(
        block_pie, request_pie
    ).resolve_scale(
        theta='independent'
    ).properties(
        title=f"Cache Hit/Miss Distribution - {data.get('cache_type', 'Unknown')} Cache (QPS: {data.get('qps', 'Unknown')})"
    )
    
    # Create combined time series chart if available
    combined_time_series = None
    if block_time_series and request_time_series:
        combined_time_series = alt.vconcat(
            block_time_series, request_time_series
        ).resolve_scale(
            color='independent'
        ).properties(
            title=f"Cache Activity Over Time - {data.get('cache_type', 'Unknown')} Cache (QPS: {data.get('qps', 'Unknown')})"
        )
    
    # Save the charts if output directory is provided
    if output_dir:
        # Create base output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Determine subdirectory name
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            qps = data.get('qps', 'unknown')
            cache_type = data.get('cache_type', 'unknown')
            subdir_name = f"cache_telemetry_{cache_type}_{qps}qps_{timestamp}"
        else:
            # Use filename as subdirectory name
            subdir_name = os.path.splitext(os.path.basename(filename))[0]
        
        # Create subdirectory
        subdir_path = os.path.join(output_dir, subdir_name)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        
        # Save pie chart
        pie_svg_path = os.path.join(subdir_path, "pie_charts.svg")
        pie_png_path = os.path.join(subdir_path, "pie_charts.png")
        combined_pie.save(pie_svg_path)
        combined_pie.save(pie_png_path)
        print(f"Pie charts saved as SVG: {pie_svg_path}")
        print(f"Pie charts saved as PNG: {pie_png_path}")
        
        # Save time series chart if available
        if combined_time_series:
            ts_svg_path = os.path.join(subdir_path, "time_series.svg")
            ts_png_path = os.path.join(subdir_path, "time_series.png")
            combined_time_series.save(ts_svg_path)
            combined_time_series.save(ts_png_path)
            print(f"Time series charts saved as SVG: {ts_svg_path}")
            print(f"Time series charts saved as PNG: {ts_png_path}")
    
    return combined_pie, combined_time_series

def process_multiple_files(file_patterns, output_dir):
    """Process multiple telemetry files matching patterns."""
    all_files = []
    
    # Handle both single string and list of patterns
    patterns = file_patterns if isinstance(file_patterns, list) else [file_patterns]
    
    # Process each pattern
    for pattern in patterns:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(pattern):
            pattern = os.path.abspath(pattern)
        
        # Check if it's a direct file path (no wildcards)
        if '*' not in pattern and '?' not in pattern and os.path.isfile(pattern):
            all_files.append(pattern)
        else:
            # It's a pattern, use glob
            matched_files = glob.glob(pattern)
            if matched_files:
                all_files.extend(matched_files)
            else:
                print(f"No files found matching pattern: {pattern}")
                # Try with current working directory if no matches
                cwd_pattern = os.path.join(os.getcwd(), pattern.lstrip('/'))
                cwd_matched_files = glob.glob(cwd_pattern)
                if cwd_matched_files:
                    all_files.extend(cwd_matched_files)
                    print(f"Found files using current directory: {cwd_pattern}")
    
    if not all_files:
        print("No files found to process")
        return
    
    print(f"Found {len(all_files)} files to process:")
    for file in all_files:
        print(f"  - {file}")
    
    for file_path in all_files:
        try:
            data = load_telemetry_data(file_path)
            base_filename = os.path.basename(file_path)
            output_filename = f"{base_filename}_viz"  # No extension, will be added by visualize_telemetry
            visualize_telemetry(data, output_dir, output_filename)
            print(f"Processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize cache telemetry data')
    parser.add_argument('--file', '-f', help='Path to the telemetry JSON file')
    parser.add_argument('--pattern', '-p', nargs='+', help='File pattern(s) or list of files to process')
    parser.add_argument('--output', '-o', default='./visualizations', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if args.pattern:
        process_multiple_files(args.pattern, args.output)
    elif args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        
        data = load_telemetry_data(args.file)
        visualize_telemetry(data, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
