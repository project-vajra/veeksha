#!/usr/bin/env python3
import json
import altair as alt
import pandas as pd
import os
import argparse
import re
import glob
import numpy as np

# Set Altair's max rows limit to prevent browser crashes
alt.data_transformers.disable_max_rows()

def downsample_timeseries(data, max_points=1000000000000000):
    """Downsample time series data to reduce memory usage."""
    if len(data) <= max_points:
        return data
    
    # Calculate the number of points to skip
    skip = len(data) // max_points
    
    # Return downsampled data
    return data[::skip]

def load_telemetry_data(file_path, file_path_ts):
    """Load telemetry data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    with open(file_path_ts, 'r') as f:
        data_ts = json.load(f)
        
    return data, data_ts

def create_block_level_charts(data, data_ts, max_points=500):
    """Create block level charts from telemetry data."""
    # Extract block level data
    block_level = data['block_level']
    
    # Create a DataFrame for the hit/miss pie chart
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
    
    # Create a DataFrame for the device/host hits pie chart
    if 'device_hits' in block_level and 'host_hits' in block_level:
        device_host_data = pd.DataFrame({
            'category': ['Device Hits', 'Host Hits'],
            'value': [block_level['device_hits'], block_level['host_hits']]
        })

        # Create pie chart for device/host hit distribution
        device_host_pie = alt.Chart(device_host_data).mark_arc().encode(
            theta=alt.Theta(field="value", type="quantitative"),
            color=alt.Color(field="category", type="nominal", 
                           scale=alt.Scale(domain=['Device Hits', 'Host Hits'], 
                                          range=['#9b59b6', '#f39c12'])),
            tooltip=['category', 'value']
        ).properties(
            title=f"Device vs Host Hits (Device: {block_level.get('device_hit_rate', 0):.2%}, Host: {block_level.get('host_hit_rate', 0):.2%})",
            width=300,
            height=300
        )
    else:
        device_host_pie = None
    
    # Create time series data for block level
    ts_data = []
    
    # Process time series data
    cumulative_total = 0
    cumulative_hits = 0
    cumulative_misses = 0
    cumulative_device_hits = 0
    cumulative_host_hits = 0
    
    # Sort by timestamp to ensure correct cumulative calculation
    total_blocks_ts = sorted(data_ts['block_level_ts'].get('total_blocks', []), key=lambda x: x[0])
    hits_ts = sorted(data_ts['block_level_ts'].get('hits', []), key=lambda x: x[0])
    misses_ts = sorted(data_ts['block_level_ts'].get('misses', []), key=lambda x: x[0])
    device_hits_ts = sorted(data_ts['block_level_ts'].get('device_hits', []), key=lambda x: x[0]) if 'device_hits' in data_ts.get('block_level_ts', {}) else []
    host_hits_ts = sorted(data_ts['block_level_ts'].get('host_hits', []), key=lambda x: x[0]) if 'host_hits' in data_ts.get('block_level_ts', {}) else []

    # print lengths of the time series
    print(f"total_blocks_ts: {len(total_blocks_ts)}")
    print(f"hits_ts: {len(hits_ts)}")
    print(f"misses_ts: {len(misses_ts)}")
    print(f"device_hits_ts: {len(device_hits_ts)}")
    print(f"host_hits_ts: {len(host_hits_ts)}")
    
    # Initialize lists for cumulative time series
    cumulative_total_ts = []
    cumulative_hits_ts = []
    cumulative_misses_ts = []
    cumulative_device_hits_ts = []
    cumulative_host_hits_ts = []

    # Process total blocks (cumulative)
    for timestamp, value in total_blocks_ts:
        cumulative_total += value
        cumulative_total_ts.append((timestamp, cumulative_total))

    # Process hits (cumulative)
    for timestamp, value in hits_ts:
        cumulative_hits += value
        cumulative_hits_ts.append((timestamp, cumulative_hits))

    # Process misses (cumulative)
    for timestamp, value in misses_ts:
        cumulative_misses += value
        cumulative_misses_ts.append((timestamp, cumulative_misses))

    # Process device hits (cumulative)
    for timestamp, value in device_hits_ts:
        cumulative_device_hits += value
        cumulative_device_hits_ts.append((timestamp, cumulative_device_hits))

    # Process host hits (cumulative)
    for timestamp, value in host_hits_ts:
        cumulative_host_hits += value
        cumulative_host_hits_ts.append((timestamp, cumulative_host_hits))

    # Downsample the cumulative time series
    downsampled_total_ts = downsample_timeseries(cumulative_total_ts, max_points)
    downsampled_hits_ts = downsample_timeseries(cumulative_hits_ts, max_points)
    downsampled_misses_ts = downsample_timeseries(cumulative_misses_ts, max_points)
    downsampled_device_hits_ts = downsample_timeseries(cumulative_device_hits_ts, max_points)
    downsampled_host_hits_ts = downsample_timeseries(cumulative_host_hits_ts, max_points)

    # Populate ts_data with downsampled cumulative values
    for timestamp, value in downsampled_total_ts:
        ts_data.append({
            'timestamp': timestamp,
            'value': value,
            'series': 'Total Blocks'
        })

    for timestamp, value in downsampled_hits_ts:
        ts_data.append({
            'timestamp': timestamp,
            'value': value,
            'series': 'Cumulative Hits'
        })

    for timestamp, value in downsampled_misses_ts:
        ts_data.append({
            'timestamp': timestamp,
            'value': value,
            'series': 'Cumulative Misses'
        })

    for timestamp, value in downsampled_device_hits_ts:
        ts_data.append({
            'timestamp': timestamp,
            'value': value,
            'series': 'Device Hits'
        })

    for timestamp, value in downsampled_host_hits_ts:
        ts_data.append({
            'timestamp': timestamp,
            'value': value,
            'series': 'Host Hits'
        })
    
    if not ts_data:
        if device_host_pie:
            return pie_chart, device_host_pie, None
        return pie_chart, None
    
    ts_df = pd.DataFrame(ts_data)
    
    # Create time series chart with cumulative values
    series_domain = ['Total Blocks', 'Cumulative Hits', 'Cumulative Misses']
    series_range = ['#3498db', '#2ecc71', '#e74c3c']
    series_dash = [[1, 0], [2, 2], [5, 2]]
    series_width = [3, 2, 2]
    
    # Add device and host hits to the domain and ranges if they exist
    if device_hits_ts:
        series_domain.append('Device Hits')
        series_range.append('#9b59b6')  # Purple for device hits
        series_dash.append([3, 3])
        series_width.append(2)
    
    if host_hits_ts:
        series_domain.append('Host Hits')
        series_range.append('#f39c12')  # Orange for host hits
        series_dash.append([1, 1])
        series_width.append(2)
    
    # Create line chart with or without points based on parameter
    time_series = alt.Chart(ts_df).encode(
        x=alt.X('timestamp:Q', title='Time (seconds)'),
        y=alt.Y('value:Q', title='Blocks'),
        color=alt.Color('series:N', 
                       scale=alt.Scale(domain=series_domain, 
                                      range=series_range)),
        tooltip=['timestamp', 'value', 'series'],
        strokeDash=alt.StrokeDash('series:N', 
                                 scale=alt.Scale(domain=series_domain,
                                               range=series_dash)),
        strokeWidth=alt.StrokeWidth('series:N',
                                   scale=alt.Scale(domain=series_domain,
                                                 range=series_width))
    ).mark_line(interpolate='basis', point=False).properties(
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
    
    if device_host_pie:
        return pie_chart, device_host_pie, time_series
    return pie_chart, time_series

def create_request_level_charts(data, data_ts, max_points=500):
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
    unique_requests_ts = sorted(data_ts['request_level_ts'].get('unique_requests', []), key=lambda x: x[0])
    hits_ts = sorted(data_ts['request_level_ts'].get('hits', []), key=lambda x: x[0])
    misses_ts = sorted(data_ts['request_level_ts'].get('misses', []), key=lambda x: x[0])
    
    # Downsample time series data if needed
    #unique_requests_ts = downsample_timeseries(unique_requests_ts, max_points)
    #hits_ts = downsample_timeseries(hits_ts, max_points)
    #misses_ts = downsample_timeseries(misses_ts, max_points)
    
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
    
    # Create line chart with or without points based on parameter
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
    ).mark_line(interpolate='basis', point=False).properties(
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

def visualize_telemetry(data, data_ts, output_dir=None, filename=None, max_points=500, low_memory_mode=False):
    """Create and save visualizations for telemetry data."""
    # Create charts
    block_charts = create_block_level_charts(data, data_ts, max_points)
    
    if len(block_charts) == 3:
        block_pie, device_host_pie, block_time_series = block_charts
    else:
        block_pie, block_time_series = block_charts
        device_host_pie = None
        
    request_pie, request_time_series = create_request_level_charts(data, data_ts, max_points)
    
    # Create combined pie chart
    if device_host_pie:
        combined_pie = alt.hconcat(
            block_pie, device_host_pie, request_pie
        ).resolve_scale(
            theta='independent'
        ).properties(
            title=f"Cache Hit/Miss Distribution - {data.get('cache_type', 'Unknown')} Cache (QPS: {data.get('qps', 'Unknown')})"
        )
    else:
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
            color='independent',
            strokeDash='independent',
            strokeWidth='independent'
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
            qps = data.get('qps', 'unknown')
            cache_type = data.get('cache_type', 'unknown')
            subdir_name = f"cache_telemetry_{cache_type}_{qps}qps"
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
        
        # Use JSON format for low memory mode
        if low_memory_mode:
            # Save as JSON to reduce memory usage
            combined_pie.save(pie_svg_path.replace('.svg', '.json'))
            print(f"Pie charts saved as JSON: {pie_svg_path.replace('.svg', '.json')}")
        else:
            #combined_pie.save(pie_svg_path)
            combined_pie.save(pie_png_path)
            print(f"Pie charts saved as SVG: {pie_svg_path}")
            print(f"Pie charts saved as PNG: {pie_png_path}")
        
        # Save time series chart if available
        if combined_time_series:
            ts_svg_path = os.path.join(subdir_path, "time_series.svg")
            ts_png_path = os.path.join(subdir_path, "time_series.png")
            
            # Use JSON format for low memory mode
            if low_memory_mode:
                # Save as JSON to reduce memory usage
                combined_time_series.save(ts_svg_path.replace('.svg', '.json'))
                print(f"Time series charts saved as JSON: {ts_svg_path.replace('.svg', '.json')}")
            else:
                #combined_time_series.save(ts_svg_path)
                combined_time_series.save(ts_png_path)
                print(f"Time series charts saved as SVG: {ts_svg_path}")
                print(f"Time series charts saved as PNG: {ts_png_path}")
    
    return combined_pie, combined_time_series

def find_telemetry_file_pairs(directory):
    """
    Find pairs of telemetry files and their corresponding timestamp files with the same QPS value.
    
    Returns a dictionary where keys are QPS values and values are lists of tuples (telemetry_file, timestamp_file)
    """
    # Regular expressions to match telemetry files and extract QPS values
    telemetry_pattern = re.compile(r'cache_telemetry_qps_(\d+\.\d+)\.json')
    ts_pattern = re.compile(r'cache_telemetry_ts_qps_(\d+\.\d+)\.json')
    
    # Find all telemetry files and timestamp files
    telemetry_files = {}
    ts_files = {}
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue
            
        # Check if it's a telemetry file
        telemetry_match = telemetry_pattern.match(filename)
        if telemetry_match:
            qps = telemetry_match.group(1)
            if qps not in telemetry_files:
                telemetry_files[qps] = []
            telemetry_files[qps].append(filepath)
            continue
            
        # Check if it's a timestamp file
        ts_match = ts_pattern.match(filename)
        if ts_match:
            qps = ts_match.group(1)
            if qps not in ts_files:
                ts_files[qps] = []
            ts_files[qps].append(filepath)
    
    # Match telemetry files with their corresponding timestamp files
    file_pairs = {}
    for qps in telemetry_files:
        if qps in ts_files:
            # Sort files to ensure consistent pairing
            telemetry_files[qps].sort()
            ts_files[qps].sort()
            
            # If there's a mismatch in the number of files, only pair what we can
            pairs = []
            for i in range(min(len(telemetry_files[qps]), len(ts_files[qps]))):
                pairs.append((telemetry_files[qps][i], ts_files[qps][i]))
            
            if pairs:
                file_pairs[qps] = pairs
    
    return file_pairs

def process_directory(directory, output_dir, max_points=500, low_memory_mode=False):
    """Process all telemetry files in a directory and generate visualizations."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Find pairs of telemetry files and timestamp files
    file_pairs = find_telemetry_file_pairs(directory)
    
    if not file_pairs:
        print(f"No matching telemetry file pairs found in {directory}")
        return
    
    # Process each pair of files
    for qps, pairs in file_pairs.items():
        print(f"Processing {len(pairs)} file pairs for QPS {qps}...")
        
        for telemetry_file, ts_file in pairs:
            try:
                print(f"Processing {os.path.basename(telemetry_file)} and {os.path.basename(ts_file)}")
                data, data_ts = load_telemetry_data(telemetry_file, ts_file)
                visualize_telemetry(data, data_ts, output_dir, telemetry_file, max_points, low_memory_mode)
            except Exception as e:
                print(f"Error processing {telemetry_file} and {ts_file}: {e}")

def process_all_sgl_cache_directories(base_directory, output_dir, max_points=500, low_memory_mode=False):
    """Process all directories that start with 'sgl-cache' in the base directory."""
    # Find all sgl-cache directories
    sgl_cache_dirs = []
    
    # Check if the base directory exists
    if not os.path.exists(base_directory):
        print(f"Base directory not found: {base_directory}")
        return sgl_cache_dirs
    
    # Find all directories that start with 'sgl-cache'
    for item in os.listdir(base_directory):
        item_path = os.path.join(base_directory, item)
        if os.path.isdir(item_path) and item.startswith('sgl-cache'):
            sgl_cache_dirs.append(item_path)
    
    if not sgl_cache_dirs:
        print(f"No directories starting with 'sgl-cache' found in {base_directory}")
        return
    
    print(f"Found {len(sgl_cache_dirs)} sgl-cache directories")
    
    # Process each directory
    for directory in sgl_cache_dirs:
        print(f"\nProcessing directory: {os.path.basename(directory)}")
        process_directory(directory, output_dir, max_points, low_memory_mode)

def collect_aggregate_data(base_directory, target_directory=None):
    """Collect aggregate data (QPS, hit rates) from telemetry files.

    If target_directory is provided, only collects data from that directory.
    Otherwise, scans base_directory for sgl-cache-* directories and collects from them.
    Calculates Host/Device hit rates relative to total blocks.
    """
    aggregate_data = []
    telemetry_pattern = re.compile(r'cache_telemetry_qps_(\d+\.\d+)\.json')
    sgl_cache_dirs = []

    if target_directory:
        # If a specific target directory is given, use only that one
        if os.path.isdir(target_directory):
            print(f"Aggregating data specifically from: {target_directory}")
            sgl_cache_dirs = [target_directory]
        else:
            print(f"Warning: Specified target directory for aggregation '{target_directory}' not found. Cannot aggregate.")
            return pd.DataFrame(aggregate_data)
    else:
        # Otherwise, scan the base_directory for sgl-cache-* dirs
        print(f"Scanning base directory '{base_directory}' for sgl-cache-* directories for aggregation.")
        if not os.path.exists(base_directory):
            print(f"Base directory not found: {base_directory}")
            return pd.DataFrame(aggregate_data)

        for item in os.listdir(base_directory):
            item_path = os.path.join(base_directory, item)
            if os.path.isdir(item_path) and item.startswith('sgl-cache'):
                sgl_cache_dirs.append(item_path)
        
        if not sgl_cache_dirs:
            print(f"No directories starting with 'sgl-cache' found in {base_directory}")
            return pd.DataFrame(aggregate_data)
        
        print(f"Found {len(sgl_cache_dirs)} sgl-cache directories for aggregation")

    # Process the selected directory/directories
    for directory in sgl_cache_dirs:
        # Use directory name directly if target_directory was specified, otherwise use basename
        dir_identifier = target_directory if target_directory else os.path.basename(directory)
        print(f"Scanning directory: {dir_identifier}") 
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                telemetry_match = telemetry_pattern.match(filename)
                if telemetry_match:
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        qps = data.get('qps')
                        block_level_data = data.get('block_level', {})
                        request_level_data = data.get('request_level', {})
                        cache_type = data.get('cache_type', 'Unknown') # Optional: Capture cache type if needed

                        # Extract block level components
                        total_blocks = block_level_data.get('total_blocks', 0)
                        host_hits = block_level_data.get('host_hits', 0)
                        device_hits = block_level_data.get('device_hits', 0)

                        # Calculate block rates relative to total blocks
                        host_block_rate = (host_hits / total_blocks) if total_blocks > 0 else 0
                        device_block_rate = (device_hits / total_blocks) if total_blocks > 0 else 0
                        # Combined block hit rate (optional, for verification)
                        # block_hit_rate = block_level_data.get('hit_rate') 

                        # Extract request level hit rate
                        request_hit_rate = request_level_data.get('hit_rate')

                        # Ensure all necessary rates are present
                        if qps is not None and request_hit_rate is not None:
                            aggregate_data.append({
                                'qps': float(qps),
                                'host_block_rate': host_block_rate,
                                'device_block_rate': device_block_rate,
                                'request_hit_rate': request_hit_rate,
                                'cache_type': cache_type,
                                'source_dir': os.path.basename(directory)
                            })
                        else:
                             print(f"Warning: Missing required data (qps, request_hit_rate) in {filename}. Skipping.")

                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {filename}. Skipping.")
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}. Skipping.")

    if not aggregate_data:
        print("No valid telemetry data found for aggregation.")
        return pd.DataFrame(aggregate_data)

    df = pd.DataFrame(aggregate_data)
    # Sort by QPS for plotting
    df = df.sort_values(by='qps').reset_index(drop=True)
    return df

def plot_aggregate_hit_rates(df, output_dir):
    """Generate and save aggregate hit rate plots (Host Block, Device Block, Request)."""
    if df.empty:
        print("Cannot plot aggregate hit rates: DataFrame is empty.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Melt the DataFrame for easier plotting with Altair
    df_melted = df.melt(
        id_vars=['qps', 'cache_type', 'source_dir'], 
        value_vars=['host_block_rate', 'device_block_rate', 'request_hit_rate'],
        var_name='level',
        value_name='hit_rate'
    )
    
    # Map level names for legend and coloring & define numeric order for stacking
    level_map = {
        'host_block_rate': {'name': 'Host Hit Rate (Block)', 'order': 1}, # Lower order stacks first
        'device_block_rate': {'name': 'Device Hit Rate (Block)', 'order': 2},
        'request_hit_rate': {'name': 'Request Hit Rate', 'order': 3} # Order only matters within stack group
    }
    # Apply mapping to get names
    df_melted['level'] = df_melted['level'].map(lambda x: level_map.get(x, {}).get('name', x))
    # Apply mapping to get numeric order - requires iterating through dict
    def get_order_num(level_name):
        for item in level_map.values():
            if item['name'] == level_name:
                return item['order']
        return 99 # Default for safety
    df_melted['stack_order_num'] = df_melted['level'].apply(get_order_num)

    # Define bar type for xOffset (grouping block components)
    def get_bar_type(level_name):
        if 'Block' in level_name:
            return 'Block Level'
        return 'Request Level'
    df_melted['bar_type'] = df_melted['level'].apply(get_bar_type)

    # Define the order for stacking within the 'Block Level' bar
    # Host hits will be at the bottom, Device hits on top
    stack_order = ['Host Hit Rate (Block)', 'Device Hit Rate (Block)']

    # Define domain for consistent coloring and legend ordering
    # Order matters for legend display
    color_domain = [level_map[k]['name'] for k in ['host_block_rate', 'device_block_rate', 'request_hit_rate']]
    color_range = ['#1f77b4', '#aec7e8', '#ff7f0e'] # Dark blue, light blue, orange

    # --- Generate descriptive title prefix from output_dir ---
    title_prefix_parts = []
    dir_lower = output_dir.lower() # Case-insensitive matching
    
    # Check for Radix type
    if 'hi' in dir_lower:
        title_prefix_parts.append("HiRadix")
    elif 'radix' in dir_lower:
        title_prefix_parts.append("Radix")
        
    # Check for Cache Policy
    if 'wts' in dir_lower:
        title_prefix_parts.append("Write Through Selective")
    elif 'wt' in dir_lower:
        title_prefix_parts.append("Write Through")
    elif 'wb' in dir_lower:
        title_prefix_parts.append("Write Back")
        
    title_prefix = " - ".join(title_prefix_parts)
    full_title = f"{title_prefix}: Cache Hit Rates by QPS" if title_prefix else "Cache Hit Rates by QPS"
    # ---

    # Create the aggregate bar chart with stacking
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('qps:O', title='QPS (Queries Per Second)', axis=alt.Axis(labelAngle=-45)),
        # Stack values using the 'zero' baseline
        y=alt.Y('hit_rate:Q', title='Hit Rate', scale=alt.Scale(domain=[0, 1]), 
                stack='zero'), 
        color=alt.Color('level:N', title='Hit Rate Type', 
                       scale=alt.Scale(domain=color_domain, range=color_range)),
        # Use bar_type to offset Block bar group from Request bar
        xOffset=alt.XOffset('bar_type:N', sort=['Block Level', 'Request Level']), 
        # Define the stacking order using the numeric helper column
        order=alt.Order('stack_order_num', sort='ascending'),
        tooltip=['qps', 'bar_type', 'level', 'hit_rate', 'cache_type', 'source_dir']
    ).properties(
        title=full_title,
        # Adjust width: Step controls space PER QPS group
        width=alt.Step(40) 
    )

    # Save the chart
    output_filename_base = os.path.join(output_dir, "aggregate_hit_rates")
    chart_json_path = f"{output_filename_base}.json"
    chart_png_path = f"{output_filename_base}.png"

    try:
        chart.save(chart_json_path)
        print(f"Aggregate hit rate chart saved as JSON: {chart_json_path}")
        chart.save(chart_png_path, scale_factor=2.0) # Higher resolution PNG
        print(f"Aggregate hit rate chart saved as PNG: {chart_png_path}")
    except Exception as e:
        print(f"Error saving aggregate chart: {e}")


def main():
    parser = argparse.ArgumentParser(description='Visualize cache telemetry data or generate aggregate reports')
    parser.add_argument('--file', '-f', help='Path to the telemetry JSON file')
    parser.add_argument('--file_ts', '-f_ts', help='Path to the telemetry JSON file with timestamps')
    parser.add_argument('--directory', '-d', help='Directory containing telemetry file pairs to process individually. Also triggers aggregate report.')
    parser.add_argument('--all-sgl-cache', '-a', action='store_true', help='Process all directories starting with sgl-cache individually')
    parser.add_argument('--aggregate', action='store_true', help='Generate aggregate plots across all sgl-cache directories found in --base-dir (runs automatically with --directory)')
    parser.add_argument('--base-dir', '-b', default='.', help='Base directory to search for sgl-cache directories (used with -a, --aggregate, or --directory)')
    parser.add_argument('--output', '-o', default='./visualizations', help='Output directory for visualizations/reports')
    parser.add_argument('--max-points', '-m', type=int, default=500, help='Maximum number of data points per time series (for individual plots)')
    parser.add_argument('--low-memory', action='store_true', help='Enable low memory mode for individual plots (saves as JSON)')
    
    args = parser.parse_args()
    
    # --- Argument Validation ---
    # Check if at least one primary mode is selected
    if not args.file and not args.directory and not args.all_sgl_cache and not args.aggregate:
        print("Error: No operation mode selected. Choose at least one of: --file, --directory, --all-sgl-cache, --aggregate")
        parser.print_help()
        return
    
    # Prevent incompatible combinations (e.g., --file and --all-sgl-cache)
    # Note: --directory now implicitly triggers --aggregate functionality later
    mutually_exclusive_modes = [args.file is not None, args.all_sgl_cache]
    if sum(mutually_exclusive_modes) > 1:
         print("Error: Cannot combine --file with --all-sgl-cache.")
         parser.print_help()
         return

    if args.file and args.directory:
        print("Error: Cannot combine --file with --directory.")
        parser.print_help()
        return

    if args.all_sgl_cache and args.directory:
        print("Error: Cannot combine --all-sgl-cache with --directory.")
        parser.print_help()
        return
        
    if args.file and not args.file_ts:
        print("Error: --file_ts must be provided when using --file")
        return

    if (args.all_sgl_cache or args.aggregate or args.directory) and args.base_dir == '.':
         print(f"Info: Using current directory ('{os.path.abspath(args.base_dir)}') as base directory for searching sgl-cache folders.")

    # --- Mode Execution ---
    max_points = args.max_points
    low_memory_mode = args.low_memory
    generate_aggregate = args.aggregate or args.directory # Determine if aggregate report is needed

    # --- Individual Processing Modes ---

    # Individual File Processing
    if args.file:
        print(f"--- Processing Single File Pair ---")
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            # Decide if we should exit or continue to aggregate if requested
            # For now, let's exit if the primary file is missing
            return 
        if not os.path.exists(args.file_ts):
            print(f"Timestamp file not found: {args.file_ts}")
            return
            
        try:
            data, data_ts = load_telemetry_data(args.file, args.file_ts)
            visualize_telemetry(data, data_ts, args.output, args.file, max_points, low_memory_mode)
            print(f"Visualizations for {os.path.basename(args.file)} saved in: {args.output}")
        except Exception as e:
            print(f"Error processing files {args.file} and {args.file_ts}: {e}")
        print(f"--- Finished Processing Single File Pair ---")

    # Individual Directory Processing
    if args.directory:
        print(f"--- Processing Directory Individually: {args.directory} ---")
        if not os.path.isdir(args.directory):
            print(f"Error: Specified directory '{args.directory}' not found or is not a directory.")
            # Continue to aggregate step if requested, but skip directory processing
        else:
            process_directory(args.directory, args.output, max_points, low_memory_mode)
        print(f"--- Finished Processing Directory ---")

    # All SGL-Cache Directory Processing (Individual)
    if args.all_sgl_cache:
        print(f"--- Processing All SGL-Cache Directories Individually (Base Dir: {args.base_dir}) ---")
        if not os.path.isdir(args.base_dir):
            print(f"Error: Base directory '{args.base_dir}' not found or is not a directory.")
            # Continue to aggregate step if requested
        else:
            process_all_sgl_cache_directories(args.base_dir, args.output, max_points, low_memory_mode)
        print(f"--- Finished Processing All SGL-Cache Directories ---")

    # --- Aggregate Report Generation ---
    # Run if --aggregate was explicitly passed OR if --directory was passed
    if generate_aggregate:
        print(f"\n--- Starting Aggregate Report Generation ---")
        # Determine the source for aggregation
        agg_target_dir = args.directory if args.directory else None
        agg_base_dir = args.base_dir

        # Pass the specific directory if provided via --directory, otherwise None
        aggregate_df = collect_aggregate_data(base_directory=agg_base_dir, target_directory=agg_target_dir)
        
        if not aggregate_df.empty:
            plot_aggregate_hit_rates(aggregate_df, args.output)
            print(f"Aggregate report generated in: {args.output}")
        else:
            print("No data collected for aggregation. Aggregate report not generated.")
        print(f"--- Finished Aggregate Report Generation ---")


if __name__ == "__main__":
    main()
