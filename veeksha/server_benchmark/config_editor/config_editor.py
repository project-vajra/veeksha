import yaml
import os
import sys
from pathlib import Path
import re

# For single keypress detection
try:
    import msvcrt  # Windows
    def getch():
        return msvcrt.getch().decode('utf-8')
except ImportError:
    try:
        import tty
        import termios
        # Unix/Linux/MacOS
        def unix_getch():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        getch = unix_getch
    except ImportError:
        # Fallback if neither method works
        def input_getch():
            return input()
        getch = input_getch

# --- Configuration ---
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_engine_configs", "vajra_config.yml")
OPTIONS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_options.yml")
MODEL_MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_mapping.yml")
OUTPUT_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment_configs"))  # Where generated .yml config files are saved
BENCHMARK_BASE_DIR = "results"  # Base directory for benchmark output specified in the config
CONFIG_EXTENSIONS = ('.yml',)

# Global variables
CONFIG_OPTIONS = None
MODEL_MAPPING = None

# --- Custom YAML Dumper ---
class ForceLiteralDumper(yaml.SafeDumper):
    """Custom YAML Dumper for multi-line strings using literal style ('|')."""
    def represent_scalar(self, tag, value, style=None):
        if isinstance(value, str) and '\n' in value:
            style = '|'
        if value is None:
             return super().represent_scalar(tag, value, style=style if style is not None else '')
        return super().represent_scalar(tag, value, style=style)

# --- Helper Functions ---

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_message(message):
    """Prints a message to the console."""
    print(message)

def load_yaml_file(filepath, file_description="config", encoding='utf-8'):
    """Loads a YAML file safely."""
    try:
        filepath = Path(filepath)
        if not filepath.is_file():
             show_message(f"Error: {file_description.capitalize()} file not found at '{filepath}'")
             return None
        with open(filepath, 'r', encoding=encoding) as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except yaml.YAMLError as e:
        show_message(f"Error loading YAML {file_description} file '{filepath}': {e}")
        if hasattr(e, 'problem_mark'):
             mark = e.problem_mark
             show_message(f"  Error position: (Line: {mark.line+1}, Column: {mark.column+1})")
        return None
    except Exception as e:
        show_message(f"Error loading {file_description} file '{filepath}': {e}")
        return None

def save_config(config_data, filename, encoding='utf-8'):
    """Saves configuration data to a YAML file using the custom dumper."""
    if not filename:
        show_message("Error: Filename cannot be empty.")
        return False, None
    if not filename.lower().endswith('.yml'):
        if filename.lower().endswith('.yaml'): filename = filename[:-5]
        filename += ".yml"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    try:
        with open(filepath, 'w', encoding=encoding) as f:
            yaml.dump(
                config_data,
                f,
                Dumper=ForceLiteralDumper,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=1000
            )
        return True, filepath
    except Exception as e:
        show_message(f"Error saving config file '{filepath}': {e}")
        return False, None

def sanitize_for_filename(value_str):
    """Sanitizes a string for path/filename use."""
    if not isinstance(value_str, str):
        value_str = str(value_str)
    sanitized = re.sub(r'[^\w\-.]+', '_', value_str)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_.')
    return sanitized or "value"

def generate_suggested_names(config_data):
    """Generates suggested dir/filenames based on config."""
    if not config_data or not isinstance(config_data, dict):
        return None, "custom_config.yml"

    server_data = config_data.get('server', {})
    model_data = config_data.get('model', {})
    parallel_data = config_data.get('parallel_spec', {})
    benchmark_data = config_data.get('benchmark_config', {})
    reqgen_data = config_data.get('request_generator_config', {})

    engine = server_data.get('openai_server_engine', 'unknown-engine')
    model_name = model_data.get('name', 'unknown-model')
    tp = parallel_data.get('tp_dimension', 'X')
    pp = parallel_data.get('pp_dimension', 'X')
    trace_path_str = reqgen_data.get('trace_request_length_generator_trace_file')
    qps_val = benchmark_data.get('qps')
    if qps_val is None:
        qps_val = reqgen_data.get('start_qps', 'X')

    engine_sanitized = sanitize_for_filename(engine)
    model_name_sanitized = sanitize_for_filename(model_name)
    tp_sanitized = sanitize_for_filename(str(tp))
    pp_sanitized = sanitize_for_filename(str(pp))

    derived_trace_name = 'no-trace'
    if trace_path_str and isinstance(trace_path_str, str):
        try:
            trace_stem = Path(trace_path_str).stem
            derived_trace_name = trace_stem.split('_')[0] if '_' in trace_stem else trace_stem
            if not derived_trace_name: derived_trace_name = 'trace'
        except Exception: pass
    elif trace_path_str:
        derived_trace_name = str(trace_path_str)
    trace_name_sanitized = sanitize_for_filename(derived_trace_name)

    qps_str = "X"
    if isinstance(qps_val, (int, float)):
        if float(qps_val) == int(qps_val): qps_str = str(int(qps_val))
        else: qps_str = str(qps_val).replace('.', '_')
    elif qps_val is not None:
        qps_str = sanitize_for_filename(str(qps_val))

    suggested_relative_dir_part = f"{engine_sanitized}_{model_name_sanitized}_tp{tp_sanitized}_pp{pp_sanitized}_{trace_name_sanitized}"
    suggested_filename = f"{engine_sanitized}_{model_name_sanitized}_tp{tp_sanitized}_pp{pp_sanitized}_{trace_name_sanitized}_qps{qps_str}.yml"

    return suggested_relative_dir_part, suggested_filename

# --- Display and Menu Functions ---

def display_main_menu(config):
    """Displays the main menu."""
    clear_screen()
    print("--- Config Editor ---")
    print("Current Configuration:")
    top_keys = []
    if not config:
        print("\n  (No configuration loaded)")
    else:
        top_keys = list(config.keys())
        for i, key in enumerate(top_keys):
            print(f"\n[{i+1}] {key}:")
            section_data = config.get(key)
            if isinstance(section_data, dict):
                if not section_data: print("    (empty)")
                else:
                    for sub_key, sub_value in section_data.items():
                        value_str = repr(sub_value); max_len = 65
                        if key == 'model' and sub_key == 'chat_template' and isinstance(sub_value, str) and len(sub_value) > max_len:
                             print(f"    {sub_key}: (long template...)")
                        else:
                            display_val = value_str[:max_len-3] + '...' if len(value_str) > max_len else value_str
                            print(f"    {sub_key}: {display_val}")
            elif isinstance(section_data, list): print(f"    (List with {len(section_data)} items)")
            elif section_data is None: print("    (None)")
            else:
                value_str = repr(section_data); max_len = 70
                display_val = value_str[:max_len-3] + '...' if len(value_str) > max_len else value_str
                print(f"    Value: {display_val}")
    print("\n\n--- Options ---")
    print("  [1..N] Edit Section | [L] Load | [S] Save | [D] Delete | [R] Reset")
    print(f"  [O] Reload Opts ({os.path.basename(OPTIONS_CONFIG_PATH)}) | [M] Reload Map ({os.path.basename(MODEL_MAPPING_PATH)}) | [Q] Quit")
    print("-" * 20)
    return top_keys

def is_editable(section_key, item_key, options_data):
    """Checks if item is editable based on options config."""
    if not options_data or not isinstance(options_data, dict): return False
    section_options = options_data.get(section_key)
    return isinstance(section_options, dict) and item_key in section_options

def display_section_menu(section_name, section_data, options_data):
    """Displays items within a section, marking editable ones."""
    clear_screen(); print(f"--- Editing Section: {section_name} ---")
    display_items = []; editable_keys_found = False
    if not isinstance(section_data, dict): print("Section data is not a dictionary.")
    else:
        current_display_index = 1
        for key, value in section_data.items():
            value_str = repr(value); max_len = 70
            if section_name == 'model' and key == 'chat_template' and isinstance(value, str) and len(value) > max_len:
                 value_display = "(long template...)"
            else:
                 value_display = value_str[:max_len-3] + '...' if len(value_str) > max_len else value_str
            is_key_editable = is_editable(section_name, key, options_data)
            marker = "*" if is_key_editable else " "
            if is_key_editable: editable_keys_found = True
            display = f" {marker}[{current_display_index}] {key}: {value_display}"
            display_items.append((current_display_index, key, display)); current_display_index += 1
        if not display_items: print("  (Section is empty)")
        else:
             for _, _, line in display_items: print(line)
    print("\n--- Options ---")
    if editable_keys_found: print("  Select number (*) to edit an item.")
    else: print("  (No items in this section are configured as editable)")
    print("  [B] Back to Main Menu")
    print("-" * 20)
    return {idx: key for idx, key, _ in display_items}

def get_single_key(prompt="", allowed_chars=None):
    """Gets a single keypress from the user without requiring Enter.
    
    Args:
        prompt: Text to display before getting input
        allowed_chars: List of allowed characters, or None to allow any
        
    Returns:
        The character pressed by the user
    """
    if prompt:
        print(prompt, end='', flush=True)
    
    while True:
        char = getch().lower()
        if allowed_chars is None or char in allowed_chars:
            print(char)  # Echo the character
            return char

def edit_section_with_options(config, section_key, options_data, model_mapping_data):
    """UI and logic for editing an item using predefined options."""
    global CONFIG_OPTIONS, MODEL_MAPPING
    if section_key not in config or not isinstance(config.get(section_key), dict):
        show_message(f"Error: Section '{section_key}' not found or is not editable.")
        return
    section_data = config[section_key]
    while True:
        current_options = CONFIG_OPTIONS if isinstance(CONFIG_OPTIONS, dict) else {}
        current_mapping = MODEL_MAPPING if isinstance(MODEL_MAPPING, dict) else {}
        key_map = display_section_menu(section_key, section_data, current_options)
        if not section_data: show_message("Section is currently empty.")
        
        # Define allowed characters based on the number of items
        allowed_chars = ['b'] + [str(i) for i in key_map.keys() if i < 10]
        
        choice = get_single_key("Enter choice (number to edit, or 'B'): ", allowed_chars)
        if choice == 'b': break
        elif choice.isdigit():
            try:
                item_index = int(choice); item_key = key_map.get(item_index)
                if item_key:
                    if not is_editable(section_key, item_key, current_options):
                        show_message(f"Item '{item_key}' is not configured as editable."); continue
                    available_options = current_options.get(section_key, {}).get(item_key, [])
                    if not isinstance(available_options, list):
                        show_message(f"Warning: Options for '{section_key}.{item_key}' not a list."); continue
                    current_value = section_data.get(item_key)
                    clear_screen(); print(f"--- Editing: {section_key}.{item_key} ---")
                    print(f"Current Value: {repr(current_value)}"); print("\nAvailable Options:")
                    if not available_options: print(f"  (No predefined options found)")
                    else:
                        for i, option in enumerate(available_options): print(f"  [{i+1}] {repr(option)}")
                    print("\n--- Actions ---")
                    if available_options: print("  Select number to choose an option.")
                    print("  [B] Back"); print("-" * 20)
                    
                    # Define allowed characters for option selection
                    option_allowed_chars = ['b'] + [str(i+1) for i in range(min(9, len(available_options)))]
                    
                    while True:
                        option_choice = get_single_key("Select option number or [B]ack: ", option_allowed_chars)
                        if option_choice == 'b': break
                        if not available_options: show_message("No options available."); break
                        if option_choice.isdigit():
                             try:
                                selected_index = int(option_choice) - 1
                                if 0 <= selected_index < len(available_options):
                                    new_value = available_options[selected_index]
                                    if new_value != current_value:
                                        section_data[item_key] = new_value
                                        show_message(f"Updated '{item_key}' to: {repr(new_value)}")
                                        if section_key == 'model' and item_key == 'name':
                                            show_message("Applying model mapping...")
                                            model_details = current_mapping.get(new_value)
                                            if model_details and isinstance(model_details, dict):
                                                id_changed, tpl_changed = False, False
                                                new_id = model_details.get('identifier')
                                                new_tpl = model_details.get('chat_template')
                                                if new_id is not None and section_data.get('identifier') != new_id:
                                                    section_data['identifier'] = new_id; id_changed = True
                                                    show_message(f"  > Updated 'identifier' to: {repr(new_id)}")
                                                if new_tpl is not None and section_data.get('chat_template') != new_tpl:
                                                     section_data['chat_template'] = new_tpl; tpl_changed = True
                                                     show_message(f"  > Updated 'chat_template' (in memory).")
                                                if not id_changed and not tpl_changed:
                                                     show_message("  > No identifier or template changes needed from mapping.")
                                            else:
                                                show_message(f"  > Warning: No mapping details found for model '{new_value}'.")
                                        show_message("Update complete.")
                                        return
                                    else: show_message("Value unchanged."); break
                                else: show_message(f"Invalid index.");
                             except ValueError: show_message("Invalid number.");
                        else: show_message("Invalid choice.");
                    continue
                else: show_message(f"Invalid selection number '{choice}'."); continue
            except ValueError: show_message("Invalid input."); continue
            except Exception as e: show_message(f"Editing error: {e}"); continue
        else: show_message("Invalid choice."); continue

# --- File Operations ---

def list_and_select_config(action_verb="load"):
    """Lists config files and prompts user for selection."""
    print(f"\n--- Select Config File to {action_verb.capitalize()} ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try: config_files = sorted([ f for f in OUTPUT_DIR.glob('*.yml') if f.is_file() ])
    except Exception as e: show_message(f"Error accessing '{OUTPUT_DIR}': {e}"); return None
    if not config_files: show_message(f"No '.yml' files found in '{OUTPUT_DIR}'."); return None
    for i, f in enumerate(config_files): print(f"  [{i+1}] {f.name}")
    print("  [B] Back"); print("-" * 20)
    
    # Define allowed characters based on the number of files
    allowed_chars = ['b'] + [str(i+1) for i in range(min(9, len(config_files)))]
    
    while True:
        choice = get_single_key(f"Enter number (or 'B'): ", allowed_chars)
        if choice == 'b': return None
        elif choice.isdigit():
            try:
                index = int(choice) - 1
                if 0 <= index < len(config_files): return config_files[index]
                else: show_message("Invalid number.")
            except ValueError: show_message("Invalid input.")
        else: show_message("Invalid choice.")

def delete_config_interactive():
    """Handles interactive deletion of a config file."""
    file_to_delete = list_and_select_config(action_verb="delete")
    if file_to_delete:
        if not file_to_delete.is_file(): show_message(f"Error: File '{file_to_delete.name}' gone.")
        else:
            confirm = get_single_key(f"Delete '{file_to_delete.name}'? (y/N): ", ['y', 'n'])
            if confirm == 'y':
                try: file_to_delete.unlink(); show_message(f"Deleted.")
                except Exception as e: show_message(f"Error deleting: {e}")
            else: show_message("Deletion cancelled.")
    else: show_message("Delete operation cancelled.")

# --- Main Execution Loop ---
def main():
    global CONFIG_OPTIONS, MODEL_MAPPING
    print("Loading initial configurations...")
    CONFIG_OPTIONS = load_yaml_file(OPTIONS_CONFIG_PATH, "options configuration")
    if CONFIG_OPTIONS is None: sys.exit(f"Fatal Error: {OPTIONS_CONFIG_PATH}")
    if not CONFIG_OPTIONS: print(f"Warning: Options file '{OPTIONS_CONFIG_PATH}' empty/invalid.")
    MODEL_MAPPING = load_yaml_file(MODEL_MAPPING_PATH, "model mapping")
    if MODEL_MAPPING is None: sys.exit(f"Fatal Error: {MODEL_MAPPING_PATH}")
    if not MODEL_MAPPING: print(f"Warning: Mapping file '{MODEL_MAPPING_PATH}' empty/invalid.")
    current_config = load_yaml_file(DEFAULT_CONFIG_PATH, "default configuration")
    if current_config is None: sys.exit(f"Fatal Error: {DEFAULT_CONFIG_PATH}")
    if not current_config: print(f"Warning: Default config '{DEFAULT_CONFIG_PATH}' empty/invalid.")
    show_message("\nInitialization complete.")

    while True:
        options = CONFIG_OPTIONS if isinstance(CONFIG_OPTIONS, dict) else {}
        mapping = MODEL_MAPPING if isinstance(MODEL_MAPPING, dict) else {}
        top_keys = display_main_menu(current_config)
        
        # Define allowed characters for main menu
        allowed_chars = ['q', 's', 'l', 'd', 'r', 'o', 'm'] + [str(i+1) for i in range(min(9, len(top_keys)))]
        
        choice = get_single_key("Enter choice: ", allowed_chars)

        if choice == 'q': break
        elif choice == 's':
            if not current_config: show_message("Cannot save empty config."); continue
            suggested_rel_dir, suggested_filename = generate_suggested_names(current_config)
            if suggested_rel_dir is None:
                 show_message("Warning: Could not generate suggested names."); suggested_rel_dir = "unknown_dir"
            if 'benchmark_config' in current_config and isinstance(current_config['benchmark_config'], dict):
                full_output_dir_str = (Path(BENCHMARK_BASE_DIR) / suggested_rel_dir).as_posix()
                if current_config['benchmark_config'].get('output_dir') != full_output_dir_str:
                    show_message(f"\nAuto-updating 'benchmark_config.output_dir': {full_output_dir_str}")
                    current_config['benchmark_config']['output_dir'] = full_output_dir_str
                    current_config['benchmark_config']['should_use_given_dir'] = True
                else:
                    show_message(f"\n'benchmark_config.output_dir' set to: {full_output_dir_str}")
                    current_config['benchmark_config']['should_use_given_dir'] = True
            else:
                show_message("\nWarning: 'benchmark_config' missing/invalid; cannot update output_dir.")

            final_filename = suggested_filename
            show_message(f"Using suggested config filename: {final_filename}")

            if final_filename:
                success, saved_filepath = save_config(current_config, final_filename)
                if success:
                    show_message(f"Config saved to '{saved_filepath}'")
            else:
                show_message("Save cancelled (error generating filename).")

        elif choice == 'l':
            selected_path = list_and_select_config(action_verb="load")
            if selected_path:
                new_cfg = load_yaml_file(selected_path, f"'{selected_path.name}'")
                if new_cfg is not None:
                    if isinstance(new_cfg, dict): current_config = new_cfg; show_message(f"Loaded '{selected_path.name}'.")
                    else: show_message(f"Error: File '{selected_path.name}' invalid structure.")
            else: show_message("Load cancelled.")
        elif choice == 'd': delete_config_interactive()
        elif choice == 'r':
             confirm = get_single_key(f"Reset config using '{DEFAULT_CONFIG_PATH}'? (y/N): ", ['y', 'n'])
             if confirm == 'y':
                 show_message(f"Reloading default..."); loaded_default = load_yaml_file(DEFAULT_CONFIG_PATH, "default")
                 if loaded_default is not None:
                      if isinstance(loaded_default, dict): current_config = loaded_default; show_message("Reset ok.")
                      else: show_message(f"Error: Default file invalid structure.")
             else: show_message("Reset cancelled.")
        elif choice == 'o':
            show_message(f"Reloading options '{OPTIONS_CONFIG_PATH}'...")
            loaded_opts = load_yaml_file(OPTIONS_CONFIG_PATH, "options");
            if loaded_opts is not None:
                if isinstance(loaded_opts, dict): CONFIG_OPTIONS = loaded_opts; show_message("Reloaded options.")
                else: show_message(f"Error: Options file invalid structure.")
        elif choice == 'm':
            show_message(f"Reloading mapping '{MODEL_MAPPING_PATH}'...")
            loaded_map = load_yaml_file(MODEL_MAPPING_PATH, "model mapping")
            if loaded_map is not None:
                if isinstance(loaded_map, dict): MODEL_MAPPING = loaded_map; show_message("Reloaded mapping.")
                else: show_message(f"Error: Mapping file invalid structure.")
        elif choice.isdigit():
            try:
                section_index = int(choice) - 1
                if 0 <= section_index < len(top_keys):
                    selected_key = top_keys[section_index]
                    current_options_state = CONFIG_OPTIONS if isinstance(CONFIG_OPTIONS, dict) else {}
                    current_mapping_state = MODEL_MAPPING if isinstance(MODEL_MAPPING, dict) else {}
                    edit_section_with_options(current_config, selected_key, current_options_state, current_mapping_state)
                else: show_message("Invalid section number.")
            except ValueError: show_message("Invalid input.")
            except Exception as e: show_message(f"Error: {e}")
        else: show_message("Invalid choice.")

    print("\nExiting config editor.")

def config_editor_entrypoint():
    # Check for essential files before starting
    if not Path(DEFAULT_CONFIG_PATH).is_file(): print(f"FATAL: Default config '{DEFAULT_CONFIG_PATH}' not found."); sys.exit(1)
    if not Path(OPTIONS_CONFIG_PATH).is_file(): print(f"FATAL: Options config '{OPTIONS_CONFIG_PATH}' not found."); sys.exit(1)
    if not Path(MODEL_MAPPING_PATH).is_file(): print(f"FATAL: Model mapping '{MODEL_MAPPING_PATH}' not found."); sys.exit(1)
    main()