from getpass import getpass
import json
import os
import subprocess
from pathlib import Path

import tqdm
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import re
import math

MAX_ARGS = 16


def decision_tree_to_str(
    tree,
    feature_names,
    multiplier,
    node=0,
    spacing=''
):
    left_children = tree.tree_.children_left
    right_children = tree.tree_.children_right
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold

    if left_children[node] == right_children[node]:
        class_counts = tree.tree_.value[node].astype(int).flatten()
        return f'{spacing}return {class_counts.argmax()}i128;'
    else:
        sub_conditions_left = decision_tree_to_str(tree, feature_names, multiplier, left_children[node], spacing + '    ')
        sub_conditions_right = decision_tree_to_str(tree, feature_names, multiplier, right_children[node], spacing + '    ')
        return f"""\
{spacing}if ({feature_names[features[node]]} <= {int(thresholds[node] * multiplier)}i128) {{
        {sub_conditions_left}
        {spacing}}}
        {spacing}else {{
        {sub_conditions_right}
        {spacing}}}"""


def split_list(lst: list, number_of_lists: int, min_elements: int = None) -> list[list]:
    elements_in_list = math.ceil(len(lst) / number_of_lists)
    if min_elements is not None:
        elements_in_list = max(elements_in_list, min_elements)
    return [lst[i * elements_in_list: (i + 1) * elements_in_list] for i in range(math.ceil(len(lst) / elements_in_list))]


def fill_group(level: int, feature_names: list[str], start_index=0):
    if len(feature_names) > MAX_ARGS:
        return {
            f'Group_{level}_{i + start_index}': fill_group(level + 1, lst, start_index=i * MAX_ARGS)
            for i, lst in enumerate(split_list(feature_names, MAX_ARGS, min_elements=MAX_ARGS))
        }
    else:
        return feature_names


def groups_to_leo_structs(groups: dict, groups_num: int, spacing=''):
    structs = []

    for key, item in groups.items():
        if isinstance(item, list):
            features_str = f'\n'.join([f'{spacing + "    "}{feature_name}: i128,' for feature_name in item])
            structs.append(f'{spacing}struct {key} {{\n{features_str}\n{spacing}}}')
        else:
            features_str = f'\n'.join(f'{spacing + "    "}{group_name.lower()}: {group_name},' for group_name in item.keys())
            structs.append(f'{spacing}struct {key} {{\n{features_str}\n{spacing}}}')
            structs.extend(groups_to_leo_structs(item, groups_num, spacing))

    return structs


def get_leo_feature_names(groups: dict) -> list:
    feature_names = []

    for key, item in groups.items():
        if isinstance(item, list):
            for feature_name in item:
                feature_names.append(f'{key.lower()}.{feature_name}')
        else:
            feature_names.extend([f'{key.lower()}.{feature_name}' for feature_name in get_leo_feature_names(item)])

    return feature_names


def groups_with_values_to_leo_input(
    groups: dict,
    original_feature_names: list,
    values: list,
    spacing='',
    recursive=False
):
    inputs = []

    for key, item in groups.items():
        if isinstance(item, list):
            ...
            group_values = [values[original_feature_names.index(feature_name)] for feature_name in item]
            group_args_str = '\n'.join([f'{spacing + "    "}{feature_name}: {int(value)}i128,' for feature_name, value in zip(item, group_values)])
            if recursive:
                inputs.append(f'{spacing}{key.lower()}: {key} {{\n{group_args_str}\n{spacing}}},')
            else:
                inputs.append(f'{spacing}{key.lower()}: {key} = {key} {{\n{group_args_str}\n{spacing}}};')
        else:
            group_args_str = '\n'.join(groups_with_values_to_leo_input(item, original_feature_names, values, spacing + '    ', recursive=True))
            if recursive:
                inputs.append(f'{spacing}{key.lower()}: {key} {{\n{group_args_str}\n{spacing}}},')
            else:
                inputs.append(f'{spacing}{key.lower()}: {key} = {key} {{\n{group_args_str}\n{spacing}}};')

    return inputs


def groups_with_values_to_run_args(
    groups: dict,
    original_feature_names: list,
    values: list,
    recursive=False
):
    args = []

    for key, item in groups.items():
        if isinstance(item, list):
            run_args = [f'{element}: {int(values[original_feature_names.index(element)])}i128' for element in item]
        else:
            run_args = groups_with_values_to_run_args(item, original_feature_names, values, recursive=True)
        run_args = ', '.join(run_args)
        args.append(f'{{ {run_args} }}')

    return args


def clear_directory(path: Path, recursive=False):
    for child in path.iterdir():
        if child.is_dir():
            clear_directory(child, recursive=True)
        else:
            child.unlink()
    if recursive:
        path.rmdir()


def main():
    possible_datasets = [
        ('iris', 'Iris dataset'),
        ('wine', 'Wine recognition'),
        ('digits', 'Digits recognition'),
        ('breast_cancer', 'Breast cancer wisconsin (diagnostic) dataset')
    ]

    for i, (_, dataset_description) in enumerate(possible_datasets):
        print(f'{i + 1}. {dataset_description}')

    while True:
        dataset_index = int(input('Input dataset index: ')) - 1

        if 0 <= dataset_index < len(possible_datasets):
            break

        print('Wrong dataset index')

    dataset_name, _ = possible_datasets[dataset_index]

    dataset = getattr(datasets, f'load_{dataset_name}')()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    dcs = DecisionTreeClassifier()
    dcs.fit(X_train, y_train)

    feature_names = [
        re.sub(
            r'[^\w_]',
            '',
            feature_name.lower().replace(' ', '_')
        ) for feature_name in dataset.feature_names
    ]

    groups_num = math.log(len(feature_names), MAX_ARGS)

    if int(groups_num) == groups_num:
        groups_num = int(groups_num) - 1
    else:
        groups_num = int(groups_num)

    feature_names_groups = fill_group(0, feature_names)

    if groups_num > 0:
        structs = groups_to_leo_structs(feature_names_groups, groups_num, spacing='    ')
        leo_feature_names = get_leo_feature_names(feature_names_groups)
    else:
        structs = []
        leo_feature_names = feature_names

    min_possible_threshold = -(2 ** 64 - 1)
    max_possible_threshold = 2 ** 64 - 1

    min_multiplier = min_possible_threshold / dcs.tree_.threshold.min()
    max_multiplier = max_possible_threshold / dcs.tree_.threshold.max()

    multiplier = min(min_multiplier, max_multiplier)

    project_path = Path(__file__).parents[1]

    leo_project_path = project_path / dataset_name

    if leo_project_path.exists():
        clear_directory(leo_project_path)
    else:
        leo_project_path.mkdir()

    build_dir = leo_project_path / 'build'
    build_dir.mkdir()

    inputs_dir = leo_project_path / 'inputs'
    inputs_dir.mkdir()

    if structs:
        input_args_str = '\n'.join(groups_with_values_to_leo_input(feature_names_groups, feature_names, [value * multiplier for value in X_test[0]]))
    else:
        input_args_str = ''
        for i, (feature_name, value) in enumerate(zip(feature_names, X_test[0])):
            input_args_str += f'{feature_name}: i128 = {int(value * multiplier)}i128;\n'

    with open(inputs_dir / f'{dataset_name}.in', 'w') as file:
        file.write(f'[predict]\n{input_args_str.strip()}')

    src_dir = leo_project_path / 'src'
    src_dir.mkdir()

    if structs:
        transition_args_str = ', '.join([f'{key.lower()}: {key}' for key in feature_names_groups.keys()])
    else:
        transition_args_str = ', '.join([f'{feature_name}: i128' for feature_name in feature_names])

    structs_str = '\n\n'.join(structs)

    if structs_str:
        structs_str += '\n\n'

    with open(src_dir / f'main.leo', 'w') as file:
        file.write(f"""\
program {dataset_name}.aleo {{
{structs_str}    transition predict({transition_args_str}) -> i128 {{
        {decision_tree_to_str(dcs, leo_feature_names, multiplier=multiplier, spacing='')}
    }}
}}""")

    private_key = input('Input private key: ')

    with open(leo_project_path / '.env', 'w') as file:
        file.write(f'NETWORK=testnet3\nPRIVATE_KEY={private_key}')

    with open(leo_project_path / '.gitignore', 'w') as file:
        file.write('.env\n*.avm\n*.prover\n*.verifier\noutputs/')

    with open(leo_project_path / 'program.json', 'w') as file:
        json.dump(
            {
                'program': f'{dataset_name}.aleo',
                'version': '0.0.0',
                'description': 'Created with Python Transpiler',
                'license': 'MIT'
            },
            file,
            indent=4
        )

    with open(Path(__file__).parent / 'README.md') as my_readme_file:
        with open(leo_project_path / 'README.md', 'w') as project_readme_file:
            project_readme_file.write(my_readme_file.read().replace('REPLACE_ME', dataset_name))

    answer = input('Do you want to evaluate the model? (y/n): ')

    if answer.lower().strip() != 'y':
        return

    y_pred = []

    for test_values in tqdm.tqdm(X_test):
        run_values = [int(value * multiplier) for value in test_values]

        if structs:
            run_args = groups_with_values_to_run_args(feature_names_groups, feature_names, run_values)
            run_args = [f'"{arg}"' for arg in run_args]
            run_args_str = ' '.join(run_args)
        else:
            run_args_str = ' '.join([f'{value}i128' for value in run_values])

        curdir = Path.cwd()
        os.chdir(leo_project_path)
        process = subprocess.Popen(
            f'leo run predict {run_args_str}',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        os.chdir(curdir)

        stdout, stderr = process.communicate()

        try:
            result = int(re.search(r'• (?P<result>\d+)i128', stdout.decode('utf-8')).group('result'))
        except Exception as e:
            print(f'Error while parsing result: {e}')
            print(f'Stdout: {stdout}')
            print(f'Stderr: {stderr}')
            print(f'Args: {run_args_str}')
            return

        y_pred.append(result)

    print('True Label Classification Report:')
    print(classification_report(y_test, y_pred, digits=3))

    print('Python Model Comparison Classification Report:')
    print(classification_report(dcs.predict(X_test), y_pred, digits=3))


if __name__ == '__main__':
    main()
