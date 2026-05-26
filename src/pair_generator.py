from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG
# =====================================================

RANDOM_STATE = 42

N_PERSONS = 100

REAL_DIR = Path("data/processed/Real")

ALTERED_DIRS = [
    Path("data/processed/Altered-Easy"),
    Path("data/processed/Altered-Medium"),
    Path("data/processed/Altered-Hard")
]

random.seed(RANDOM_STATE)

# =====================================================
# HELPERS
# =====================================================

def get_person_id(filename):

    return filename.split("__")[0]


def get_identity(filename):

    name = filename.replace(".BMP", "")

    parts = name.split("__")

    person_id = parts[0]

    rest = parts[1]

    for suffix in [
        "_CR",
        "_Obl",
        "_Zcut"
    ]:
        rest = rest.replace(
            suffix,
            ""
        )

    return f"{person_id}__{rest}"


def get_finger_type(identity):

    finger = identity.split("__")[1]

    parts = finger.split("_")

    return "_".join(parts[1:])


# =====================================================
# LOAD REAL FILES
# =====================================================

real_files = sorted([
    f.name
    for f in REAL_DIR.glob("*.BMP")
])

# =====================================================
# SAMPLE PERSONS
# =====================================================

all_persons = sorted(
    list(
        set(
            get_person_id(f)
            for f in real_files
        )
    )
)

selected_persons = random.sample(
    all_persons,
    N_PERSONS
)

# =====================================================
# SPLIT PERSONS
# =====================================================

train_persons, temp_persons = train_test_split(
    selected_persons,
    test_size=0.30,
    random_state=RANDOM_STATE
)

valid_persons, test_persons = train_test_split(
    temp_persons,
    test_size=0.50,
    random_state=RANDOM_STATE
)

# =====================================================
# BUILD POSITIVE PAIRS
# =====================================================

def build_positive_pairs(persons):

    real_map = {}

    for file in real_files:

        if get_person_id(file) not in persons:
            continue

        identity = get_identity(file)

        real_map[identity] = file

    positive_pairs = []

    for altered_dir in ALTERED_DIRS:

        for file in altered_dir.glob("*.BMP"):

            altered_name = file.name

            if get_person_id(
                altered_name
            ) not in persons:
                continue

            identity = get_identity(
                altered_name
            )

            if identity not in real_map:
                continue

            positive_pairs.append([
                f"Real/{real_map[identity]}",
                f"{altered_dir.name}/{altered_name}",
                1
            ])

    return positive_pairs, real_map


# =====================================================
# BUILD HARD NEGATIVE PAIRS
# =====================================================

def build_negative_pairs(
    real_map,
    n_pairs
):

    finger_groups = {}

    for identity in real_map:

        finger_type = get_finger_type(
            identity
        )

        finger_groups.setdefault(
            finger_type,
            []
        ).append(identity)

    negative_pairs = []

    max_trials = n_pairs * 20

    trials = 0

    while (
        len(negative_pairs) < n_pairs
        and
        trials < max_trials
    ):

        trials += 1

        finger_type = random.choice(
            list(
                finger_groups.keys()
            )
        )

        candidates = finger_groups[
            finger_type
        ]

        if len(candidates) < 2:
            continue

        id1, id2 = random.sample(
            candidates,
            2
        )

        person1 = id1.split("__")[0]
        person2 = id2.split("__")[0]

        if person1 == person2:
            continue

        negative_pairs.append([
            f"Real/{real_map[id1]}",
            f"Real/{real_map[id2]}",
            0
        ])

    return negative_pairs


# =====================================================
# BUILD DATASET
# =====================================================

def build_dataset(persons):

    positive_pairs, real_map = (
        build_positive_pairs(
            persons
        )
    )

    negative_pairs = (
        build_negative_pairs(
            real_map,
            len(positive_pairs)
        )
    )

    df = pd.DataFrame(
        positive_pairs +
        negative_pairs,
        columns=[
            "img1",
            "img2",
            "label"
        ]
    )

    return df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )


# =====================================================
# CREATE SPLITS
# =====================================================

train_df = build_dataset(
    train_persons
)

valid_df = build_dataset(
    valid_persons
)

test_df = build_dataset(
    test_persons
)

# =====================================================
# SAVE
# =====================================================

Path("data").mkdir(
    exist_ok=True
)

train_df.to_csv(
    "data/pairs_train.csv",
    index=False
)

valid_df.to_csv(
    "data/pairs_valid.csv",
    index=False
)

test_df.to_csv(
    "data/pairs_test.csv",
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n==========")
print("PAIRS CREATED")
print("==========")

print(
    f"Selected persons : {len(selected_persons)}"
)

print(
    f"Train persons    : {len(train_persons)}"
)

print(
    f"Valid persons    : {len(valid_persons)}"
)

print(
    f"Test persons     : {len(test_persons)}"
)

print()

print(
    f"Train Pairs : {len(train_df):,}"
)

print(
    f"Valid Pairs : {len(valid_df):,}"
)

print(
    f"Test Pairs  : {len(test_df):,}"
)

print()

print(
    f"Train Positive : {(train_df.label==1).sum():,}"
)

print(
    f"Train Negative : {(train_df.label==0).sum():,}"
)

print(
    train_df[
        train_df.label == 0
    ].head(20)
)