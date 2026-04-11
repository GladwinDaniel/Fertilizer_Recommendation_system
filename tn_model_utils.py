from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
import requests
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
import re
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType


DATA_FILE = Path(__file__).resolve().parent / "datasets" / "fertilizer_recommendation.csv"
MODEL_DIR = Path(__file__).resolve().parent / "artifacts" / "tn_fertilizer_spark_model"
MODEL_METADATA_FILE = MODEL_DIR / "metadata.json"
MODEL_ARTIFACT_PATH = MODEL_DIR

TN_CENTER = (11.1271, 78.6569)
TN_DISTRICTS = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Erode": (11.3410, 77.7172),
    "Thanjavur": (10.7870, 79.1378),
    "Tirunelveli": (8.7139, 77.7567),
    "Dindigul": (10.3673, 77.9803),
    "Vellore": (12.9165, 79.1325),
    "Cuddalore": (11.7480, 79.7714),
    "Nagapattinam": (10.7656, 79.8428),
}
DISTRICT_REGION_MAP = {
    "Chennai": "North",
    "Vellore": "North",
    "Cuddalore": "East",
    "Nagapattinam": "East",
    "Thanjavur": "Central",
    "Tiruchirappalli": "Central",
    "Salem": "West",
    "Erode": "West",
    "Coimbatore": "West",
    "Madurai": "South",
    "Tirunelveli": "South",
    "Dindigul": "South",
}

TN_CROPS = ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"]
CROP_GROWTH_STAGES = ["Sowing", "Vegetative", "Flowering", "Harvest"]
IRRIGATION_TYPES = ["Rainfed", "Canal", "Drip", "Sprinkler"]
SOIL_TYPES = ["Clay", "Loamy", "Sandy", "Silt"]
REGIONS = ["North", "South", "East", "West", "Central"]

SOIL_DEFAULTS = {
    "Clay": {"soil_ph": 6.4, "soil_moisture": 34.0, "organic_carbon": 0.88, "ec": 1.08},
    "Loamy": {"soil_ph": 6.7, "soil_moisture": 28.0, "organic_carbon": 0.74, "ec": 0.82},
    "Sandy": {"soil_ph": 7.2, "soil_moisture": 22.0, "organic_carbon": 0.46, "ec": 1.18},
    "Silt": {"soil_ph": 6.5, "soil_moisture": 31.0, "organic_carbon": 0.67, "ec": 0.64},
}

REGION_RAINFALL_BASE = {
    "North": 980.0,
    "South": 820.0,
    "East": 1340.0,
    "West": 760.0,
    "Central": 1080.0,
}
SEASON_RAINFALL_ADJ = {
    "Kharif": 540.0,
    "Rabi": -120.0,
    "Zaid": 220.0,
}
SEASON_MONTH_MAP = {
    1: "Rabi",
    2: "Rabi",
    3: "Zaid",
    4: "Zaid",
    5: "Zaid",
    6: "Kharif",
    7: "Kharif",
    8: "Kharif",
    9: "Kharif",
    10: "Rabi",
    11: "Rabi",
    12: "Rabi",
}

FEATURE_COLUMNS = [
    "Soil_Type",
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Nitrogen_Level",
    "Phosphorus_Level",
    "Potassium_Level",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Region",
]
CATEGORICAL_COLUMNS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season", "Irrigation_Type", "Region"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("TamilNaduFertilizerRecommendation")
        .master("local[1]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def infer_agri_season(month: int) -> str:
    return SEASON_MONTH_MAP.get(int(month), "Kharif")


def district_to_region(district: str) -> str:
    return DISTRICT_REGION_MAP.get(district, "Central")


def soil_defaults(soil_type: str) -> dict[str, float]:
    return SOIL_DEFAULTS.get(soil_type, SOIL_DEFAULTS["Loamy"]).copy()


def load_base_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_FILE}")
    return pd.read_csv(DATA_FILE)


def load_training_data() -> pd.DataFrame:
    return load_base_data()


def _normalize_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = FEATURE_COLUMNS + ["Recommended_Fertilizer"]
    return df[required].copy()


def _estimate_rainfall(region: str, season: str, month: int) -> float:
    monthly_adj = {
        1: -90.0,
        2: -60.0,
        3: 20.0,
        4: 60.0,
        5: 110.0,
        6: 210.0,
        7: 260.0,
        8: 240.0,
        9: 180.0,
        10: 90.0,
        11: 30.0,
        12: -40.0,
    }
    rainfall = REGION_RAINFALL_BASE.get(region, 1000.0) + SEASON_RAINFALL_ADJ.get(season, 0.0) + monthly_adj.get(int(month), 0.0)
    return float(np.clip(rainfall, 350.0, 2800.0))


def _to_spark_dataframe(spark: SparkSession, pdf: pd.DataFrame) -> SparkDataFrame:
    pdf = pdf.copy()
    for column in NUMERIC_COLUMNS:
        pdf[column] = pdf[column].astype(float)
    schema = StructType(
        [
            StructField("Soil_Type", StringType(), True),
            StructField("Soil_pH", DoubleType(), True),
            StructField("Soil_Moisture", DoubleType(), True),
            StructField("Organic_Carbon", DoubleType(), True),
            StructField("Electrical_Conductivity", DoubleType(), True),
            StructField("Nitrogen_Level", DoubleType(), True),
            StructField("Phosphorus_Level", DoubleType(), True),
            StructField("Potassium_Level", DoubleType(), True),
            StructField("Temperature", DoubleType(), True),
            StructField("Humidity", DoubleType(), True),
            StructField("Rainfall", DoubleType(), True),
            StructField("Crop_Type", StringType(), True),
            StructField("Crop_Growth_Stage", StringType(), True),
            StructField("Season", StringType(), True),
            StructField("Irrigation_Type", StringType(), True),
            StructField("Region", StringType(), True),
            StructField("Recommended_Fertilizer", StringType(), True),
        ]
    )
    return spark.createDataFrame(pdf[FEATURE_COLUMNS + ["Recommended_Fertilizer"]], schema=schema)


def build_input_frame(
    soil_type: str,
    soil_ph: float,
    soil_moisture: float,
    organic_carbon: float,
    electrical_conductivity: float,
    nitrogen_level: int,
    phosphorus_level: int,
    potassium_level: int,
    temperature: float,
    humidity: float,
    rainfall: float,
    crop_type: str,
    crop_growth_stage: str,
    irrigation_type: str,
    region: str,
    selected_date: date,
) -> pd.DataFrame:
    season = infer_agri_season(selected_date.month)
    return pd.DataFrame(
        {
            "Soil_Type": [soil_type],
            "Soil_pH": [soil_ph],
            "Soil_Moisture": [soil_moisture],
            "Organic_Carbon": [organic_carbon],
            "Electrical_Conductivity": [electrical_conductivity],
            "Nitrogen_Level": [nitrogen_level],
            "Phosphorus_Level": [phosphorus_level],
            "Potassium_Level": [potassium_level],
            "Temperature": [temperature],
            "Humidity": [humidity],
            "Rainfall": [rainfall],
            "Crop_Type": [crop_type],
            "Crop_Growth_Stage": [crop_growth_stage],
            "Season": [season],
            "Irrigation_Type": [irrigation_type],
            "Region": [region],
        }
    )


def train_and_save_model(artifact_path: Path = MODEL_DIR) -> dict[str, Any]:
    spark = spark_session()
    sdf = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(DATA_FILE))
        .select(*(FEATURE_COLUMNS + ["Recommended_Fertilizer"]))
    )
    for column in NUMERIC_COLUMNS:
        sdf = sdf.withColumn(column, F.col(column).cast("double"))

    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="error")
        for col in CATEGORICAL_COLUMNS
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{col}_idx" for col in CATEGORICAL_COLUMNS],
        outputCols=[f"{col}_ohe" for col in CATEGORICAL_COLUMNS],
        dropLast=False,
        handleInvalid="error",
    )
    assembler_inputs = [f"{col}_ohe" for col in CATEGORICAL_COLUMNS] + NUMERIC_COLUMNS
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
    label_indexer = StringIndexer(inputCol="Recommended_Fertilizer", outputCol="label", handleInvalid="error")

    classifier = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        predictionCol="prediction",
        probabilityCol="probability",
        numTrees=140,
        maxDepth=16,
        featureSubsetStrategy="sqrt",
        subsamplingRate=0.9,
        impurity="gini",
        minInstancesPerNode=1,
        minInfoGain=0.0,
        seed=42,
    )

    pipeline = Pipeline(stages=indexers + [encoder, assembler, label_indexer, classifier])

    train_df, test_df = sdf.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = float(evaluator.evaluate(predictions))

    categorical_labels = {
        column: list(model.stages[index].labels)
        for index, column in enumerate(CATEGORICAL_COLUMNS)
    }
    label_stage = model.stages[len(CATEGORICAL_COLUMNS) + 2]
    classifier_stage = model.stages[-1]
    tree_debug_strings = [tree.toDebugString for tree in classifier_stage.trees]

    artifact_payload = {
        "model_kind": "spark_random_forest_json",
        "accuracy": accuracy,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_labels": categorical_labels,
        "label_names": list(label_stage.labels),
        "tree_debug_strings": tree_debug_strings,
        "num_trees": len(tree_debug_strings),
        "soil_types": SOIL_TYPES,
        "crop_types": TN_CROPS,
        "crop_growth_stages": CROP_GROWTH_STAGES,
        "irrigation_types": IRRIGATION_TYPES,
        "regions": REGIONS,
        "train_size": train_df.count(),
        "test_size": test_df.count(),
    }
    artifact_path.mkdir(parents=True, exist_ok=True)
    MODEL_METADATA_FILE.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
    return artifact_payload


def load_trained_artifacts(artifact_path: Path = MODEL_DIR) -> dict[str, Any]:
    if MODEL_METADATA_FILE.exists():
        metadata = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    else:
        metadata = {
            "model_kind": "spark_logistic_regression_json",
            "feature_columns": FEATURE_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "numeric_columns": NUMERIC_COLUMNS,
            "soil_types": SOIL_TYPES,
            "crop_types": TN_CROPS,
            "crop_growth_stages": CROP_GROWTH_STAGES,
            "irrigation_types": IRRIGATION_TYPES,
            "regions": REGIONS,
            "accuracy": 0.0,
            "train_size": 0,
            "test_size": 0,
            "categorical_labels": {},
            "label_names": [],
            "tree_debug_strings": [],
            "num_trees": 0,
        }
    metadata["model"] = metadata
    metadata["parsed_trees"] = [_parse_tree_debug_string(tree) for tree in metadata.get("tree_debug_strings", [])]
    return metadata


def _feature_vector_from_frame(features: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    row = features.iloc[0]
    vector_parts: list[np.ndarray] = []

    for column in metadata["categorical_columns"]:
        labels = list(metadata["categorical_labels"].get(column, []))
        encoded = np.zeros(len(labels), dtype=float)
        value = str(row[column])
        if value in labels:
            encoded[labels.index(value)] = 1.0
        vector_parts.append(encoded)

    for column in metadata["numeric_columns"]:
        vector_parts.append(np.array([float(row[column])], dtype=float))

    return np.concatenate(vector_parts)


def prediction_with_confidence(model: dict[str, Any], features: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    feature_vector = _feature_vector_from_frame(features, model)

    if not model.get("parsed_trees"):
        model["parsed_trees"] = [_parse_tree_debug_string(tree) for tree in model.get("tree_debug_strings", [])]

    class_labels = list(model["label_names"])
    vote_counts = np.zeros(len(class_labels), dtype=float)

    for tree in model["parsed_trees"]:
        prediction_index = int(round(_predict_tree(tree, feature_vector)))
        if 0 <= prediction_index < len(vote_counts):
            vote_counts[prediction_index] += 1.0

    if vote_counts.sum() == 0:
        vote_counts[:] = 1.0

    probability_vector = vote_counts / vote_counts.sum()
    label_index = int(np.argmax(probability_vector))
    ranked = np.argsort(probability_vector)[::-1]
    confidence_table = pd.DataFrame(
        {
            "Fertilizer": [class_labels[i] for i in ranked],
            "Confidence": np.round(probability_vector[ranked] * 100, 2),
        }
    ).head(3)
    return class_labels[label_index], confidence_table


_TREE_CONDITION_RE = re.compile(r"feature\s+(\d+)\s+(<=|<|>=|>|=|==|in|not in)\s+(.+)")


def _parse_condition(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("If "):
        cleaned = cleaned[3:].strip()
    if cleaned.startswith("Else "):
        cleaned = cleaned[5:].strip()
    cleaned = cleaned.strip("()")
    match = _TREE_CONDITION_RE.search(cleaned)
    if not match:
        raise ValueError(f"Unsupported tree condition: {text}")

    feature_index = int(match.group(1))
    operator = match.group(2)
    raw_value = match.group(3).strip()
    if operator in {"in", "not in"}:
        values = [float(part.strip()) for part in raw_value.strip("{}").split(",") if part.strip()]
        return {"feature_index": feature_index, "operator": operator, "values": values}
    return {"feature_index": feature_index, "operator": operator, "threshold": float(raw_value)}


def _parse_tree_debug_string(debug_string: str) -> dict[str, Any]:
    lines = [
        line.rstrip()
        for line in debug_string.splitlines()
        if line.strip().startswith(("If ", "Else", "Predict:"))
    ]

    def parse_node(start_index: int, parent_indent: int) -> tuple[dict[str, Any] | None, int]:
        index = start_index
        while index < len(lines) and len(lines[index]) - len(lines[index].lstrip(" ")) <= parent_indent:
            index += 1
        if index >= len(lines):
            return None, index

        line = lines[index]
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if stripped.startswith("Predict:"):
            return {"type": "leaf", "prediction": float(stripped.split(":", 1)[1].strip())}, index + 1

        if stripped.startswith("If "):
            node = {"type": "if", "condition": _parse_condition(stripped)}
            true_child, next_index = parse_node(index + 1, indent)
            node["true"] = true_child

            false_child = None
            if next_index < len(lines):
                next_line = lines[next_index]
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent == indent and next_line.strip().startswith("Else"):
                    false_child, next_index = parse_node(next_index + 1, indent)
            node["false"] = false_child
            return node, next_index

        if stripped.startswith("Else"):
            return parse_node(index + 1, parent_indent)

        return None, index + 1

    root, _ = parse_node(0, -1)
    if root is None:
        raise ValueError("Unable to parse tree debug string")
    return root


def _evaluate_tree_condition(condition: dict[str, Any], feature_vector: np.ndarray) -> bool:
    value = float(feature_vector[condition["feature_index"]])
    operator = condition["operator"]
    if operator == "<=":
        return value <= condition["threshold"]
    if operator == "<":
        return value < condition["threshold"]
    if operator == ">=":
        return value >= condition["threshold"]
    if operator == ">":
        return value > condition["threshold"]
    if operator in {"=", "=="}:
        return value == condition["threshold"]
    if operator == "in":
        return value in condition["values"]
    if operator == "not in":
        return value not in condition["values"]
    raise ValueError(f"Unsupported operator: {operator}")


def _predict_tree(node: dict[str, Any], feature_vector: np.ndarray) -> float:
    if node["type"] == "leaf":
        return float(node["prediction"])
    branch = node["true"] if _evaluate_tree_condition(node["condition"], feature_vector) else node["false"]
    if branch is None:
        return float(node["condition"]["feature_index"])
    return _predict_tree(branch, feature_vector)


def _fertilizer_profile(fertilizer_name: str) -> str:
    name = str(fertilizer_name).strip()
    if name == "Urea":
        return "strong nitrogen support"
    if name == "DAP":
        return "balanced nitrogen and phosphorous support"
    if name == "MOP":
        return "potassium support for root and yield development"
    if name == "NPK":
        return "balanced NPK support across growth stages"
    if name == "Zinc Sulphate":
        return "micronutrient support for deficiency correction"
    if name == "Compost":
        return "organic nutrient release and soil health improvement"
    if name == "SSP":
        return "phosphorous support and early root development"
    return "nutrient balance suited to the field condition"


def generate_suitability_note(
    fertilizer_name: str,
    crop_type: str,
    soil_type: str,
    soil_ph: float,
    soil_moisture: float,
    rainfall: float,
    nitrogen_level: int,
    phosphorus_level: int,
    potassium_level: int,
    crop_growth_stage: str,
    irrigation_type: str,
    region: str,
    season: str,
) -> str:
    nutrient_gaps = []
    if nitrogen_level < 55:
        nutrient_gaps.append("nitrogen")
    if phosphorus_level < 40:
        nutrient_gaps.append("phosphorous")
    if potassium_level < 40:
        nutrient_gaps.append("potassium")

    nutrient_text = ", ".join(nutrient_gaps) if nutrient_gaps else "balanced nutrient levels"
    profile = _fertilizer_profile(fertilizer_name)
    moisture_text = "good moisture retention" if soil_moisture >= 28 else "lower moisture retention"
    rainfall_text = "adequate rainfall support" if rainfall >= 900 else "moderate rainfall conditions"

    return (
        f"{fertilizer_name} is suitable for {crop_type} in {soil_type} soil because the field shows {nutrient_text}, "
        f"the soil pH is {soil_ph:.1f}, and this fertilizer provides {profile}. "
        f"It also fits {crop_growth_stage.lower()} stage management under {irrigation_type.lower()} irrigation in the {region} region with {moisture_text} and {rainfall_text} during {season}."
    )


def fetch_live_context(latitude: float, longitude: float, region: str, season: str, soil_type: str) -> dict[str, float | str]:
    soil = soil_defaults(soil_type)
    month = date.today().month
    rainfall = _estimate_rainfall(region, season, month)

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m",
            },
            timeout=8,
        )
        response.raise_for_status()
        current = response.json().get("current", {})
        temperature = float(current.get("temperature_2m", 30.0))
        humidity = float(current.get("relative_humidity_2m", 70.0))
        source = "Open-Meteo + TN heuristics"
    except Exception:
        temperature = 30.0
        humidity = 70.0
        source = "TN heuristics"

    soil_moisture = float(np.clip((humidity / 3.0) + (rainfall / 100.0), 5.0, 95.0))

    return {
        "temperature": round(float(np.clip(temperature, 12.0, 45.0)), 2),
        "humidity": round(float(np.clip(humidity, 15.0, 99.0)), 2),
        "rainfall": round(float(rainfall), 2),
        "soil_moisture": round(float(np.clip(soil_moisture, 5.0, 95.0)), 2),
        "soil_ph": soil["soil_ph"],
        "organic_carbon": soil["organic_carbon"],
        "electrical_conductivity": soil["ec"],
        "source": source,
    }
