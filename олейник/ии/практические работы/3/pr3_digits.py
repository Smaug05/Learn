"""ПР-3: обучение MLP на digits и сохранение фактических результатов.

Запуск: python pr3_digits.py
Результаты по умолчанию записываются рядом со скриптом.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

# Ограничение действует только в этом процессе и делает малый опыт быстрее.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def train_model(architecture, X_train, X_test, y_train, y_test, *, early_stopping):
    """Обучить один заданный вариант; сохранить, а не скрыть предупреждения."""
    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        max_iter=500,
        early_stopping=early_stopping,
        validation_fraction=0.15,
        n_iter_no_change=30 if early_stopping else 10,
        random_state=42,
    )
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    seconds = perf_counter() - started
    messages = [f"{type(item.message).__name__}: {item.message}" for item in caught]
    for message in messages:
        print(f"WARNING {architecture}: {message}", file=sys.stderr, flush=True)
    prediction = model.predict(X_test)
    result = {
        "architecture": list(architecture),
        "early_stopping": early_stopping,
        "epochs": int(model.n_iter_),
        "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_correct": int(np.count_nonzero(prediction == y_test)),
        "test_errors": int(np.count_nonzero(prediction != y_test)),
        "parameters": int(sum(a.size for a in model.coefs_) + sum(a.size for a in model.intercepts_)),
        "last_loss": float(model.loss_curve_[-1]),
        "fit_seconds": round(seconds, 6),
        "convergence_warning": any(isinstance(item.message, ConvergenceWarning) for item in caught),
        "warnings": messages,
    }
    print(
        f"{str(architecture):>9}: epochs={result['epochs']:3d}, "
        f"train={result['train_accuracy']:.6f}, test={result['test_accuracy']:.6f} "
        f"({result['test_correct']}/{len(y_test)})",
        flush=True,
    )
    return model, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    output = parser.parse_args().output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Изображений: {X.shape[0]}; признаков: {X.shape[1]}", flush=True)
    print(f"Обучение: {len(y_train)}; тест: {len(y_test)}; нормализация X / 16.0", flush=True)
    print(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}", flush=True)

    figure, axes = plt.subplots(2, 8, figsize=(9, 3))
    for ax, digit, label in zip(axes.ravel(), digits.images, y):
        ax.imshow(digit, cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
        ax.set_title(str(label))
        ax.axis("off")
    figure.suptitle("Примеры digits: исходные изображения 8×8")
    figure.tight_layout()
    figure.savefig(output / "digits_examples.png", dpi=160)
    plt.close(figure)

    baseline, baseline_result = train_model(
        (32,), X_train, X_test, y_train, y_test, early_stopping=True
    )
    prediction = baseline.predict(X_test)
    cm = confusion_matrix(y_test, prediction, labels=np.arange(10))
    baseline_result.update({
        "best_validation_accuracy": float(baseline.best_validation_score_),
        "first_best_validation_epoch": int(np.argmax(baseline.validation_scores_) + 1),
        "loss_curve": [float(value) for value in baseline.loss_curve_],
        "validation_scores": [float(value) for value in baseline.validation_scores_],
        "confusion_matrix": cm.tolist(),
    })
    epochs = np.arange(1, baseline.n_iter_ + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, baseline.loss_curve_, color="#23578c")
    axes[0].set(title="Функция потерь", xlabel="Эпоха", ylabel="Потери")
    axes[1].plot(epochs, baseline.validation_scores_, color="#26765f")
    axes[1].set(title="Точность на внутренней проверке", xlabel="Эпоха", ylabel="Accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
    figure.suptitle("Базовая сеть 64 → 32 → 10, early_stopping=True")
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=160)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(7.5, 6.5))
    matrix_image = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set(
        xticks=np.arange(10), yticks=np.arange(10),
        xlabel="Ответ сети", ylabel="Правильная цифра",
        title=f"Матрица ошибок базовой сети: {baseline_result['test_correct']}/{len(y_test)}",
    )
    for row in range(10):
        for column in range(10):
            ax.text(column, row, str(cm[row, column]), ha="center", va="center",
                    color="white" if cm[row, column] > cm.max() / 2 else "black")
    figure.colorbar(matrix_image, ax=ax, label="Число изображений")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    print("Матрица ошибок (строки — правильная цифра, столбцы — ответ):", flush=True)
    print(cm, flush=True)

    wrong_positions = np.flatnonzero(prediction != y_test)
    probabilities = baseline.predict_proba(X_test)
    mistakes = [{
        "test_position": int(position),
        "dataset_index": int(test_indices[position]),
        "true_digit": int(y_test[position]),
        "predicted_digit": int(prediction[position]),
        "predicted_probability": float(probabilities[position, prediction[position]]),
    } for position in wrong_positions]
    pairs = sorted(
        [{"true_digit": i, "predicted_digit": j, "count": int(cm[i, j])}
         for i in range(10) for j in range(10) if i != j and cm[i, j]],
        key=lambda item: (-item["count"], item["true_digit"], item["predicted_digit"]),
    )
    print("Ошибки цифра → ответ:", pairs, flush=True)
    if mistakes:
        columns = min(5, len(mistakes))
        rows = int(np.ceil(len(mistakes) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(2.25 * columns, 3.2 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, mistake in zip(axes.ravel(), mistakes):
            ax.imshow(digits.images[mistake["dataset_index"]], cmap="gray_r",
                      vmin=0, vmax=16, interpolation="nearest")
            ax.set_title(
                f"{mistake['true_digit']} → {mistake['predicted_digit']}\n"
                f"№{mistake['dataset_index']}; p={mistake['predicted_probability']:.2f}"
            )
        figure.suptitle("Все ошибки базовой сети: правильная цифра → ответ")
        figure.subplots_adjust(top=0.84, bottom=0.04, wspace=0.25, hspace=0.55)
    else:
        figure, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "Ошибок на тесте нет", ha="center", va="center")
        ax.axis("off")
    figure.savefig(output / "misclassified_digits.png", dpi=160)
    plt.close(figure)
    with (output / "misclassified_digits.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "test_position", "dataset_index", "true_digit", "predicted_digit", "predicted_probability"
        ])
        writer.writeheader()
        writer.writerows(mistakes)

    # Четыре обязательных варианта и выбранное самостоятельное задание.
    experiments = []
    models = []
    for architecture in [(2,), (8,), (32,), (128,), (16, 16), (64, 32)]:
        model, result = train_model(
            architecture, X_train, X_test, y_train, y_test, early_stopping=False
        )
        experiments.append(result)
        models.append(model)

    # Ровно заданные шесть вариантов, без поиска дополнительных настроек.
    # При равной точности выбирается модель с меньшим числом параметров.
    best_index = min(range(len(experiments)), key=lambda index: (
        -experiments[index]["test_accuracy"], experiments[index]["parameters"], index
    ))
    stage5_best_index = min(range(4), key=lambda index: (
        -experiments[index]["test_accuracy"], experiments[index]["parameters"], index
    ))
    joblib.dump(models[best_index], output / "digits_net.joblib")
    restored = joblib.load(output / "digits_net.joblib")
    if not np.array_equal(restored.predict(X_test), models[best_index].predict(X_test)):
        raise RuntimeError("Предсказания загруженной модели отличаются")

    csv_fields = ["architecture", "early_stopping", "epochs", "train_accuracy", "test_accuracy",
                  "test_correct", "test_errors", "parameters", "last_loss", "fit_seconds", "convergence_warning"]
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(experiments)

    figure, ax = plt.subplots(figsize=(9, 4.3))
    positions = np.arange(len(experiments))
    ax.bar(positions - 0.18, [row["train_accuracy"] for row in experiments], width=0.36,
           color="#23578c", label="Обучение")
    ax.bar(positions + 0.18, [row["test_accuracy"] for row in experiments], width=0.36,
           color="#7ab5a3", label="Тест")
    ax.set(xticks=positions, xticklabels=[str(tuple(row["architecture"])) for row in experiments],
           ylim=(0.45, 1.03), xlabel="Скрытые слои", ylabel="Accuracy",
           title="Размер сети: без ранней остановки, max_iter=500")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output / "network_comparison.png", dpi=160)
    plt.close(figure)

    results = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__, "joblib": joblib.__version__},
        "dataset": {"name": "sklearn.datasets.load_digits", "samples": len(y), "features": X.shape[1],
                    "normalization": "digits.data / 16.0", "train_samples": len(y_train),
                    "test_samples": len(y_test), "test_size": 0.2, "random_state": 42,
                    "stratify": True, "internal_validation_fraction": 0.15,
                    "internal_validation_samples": int(np.ceil(len(y_train) * 0.15)),
                    "train_class_counts": np.bincount(y_train, minlength=10).tolist(),
                    "test_class_counts": np.bincount(y_test, minlength=10).tolist()},
        "baseline": baseline_result,
        "confusion_pairs": pairs,
        "mistakes": mistakes,
        "experiments": experiments,
        "stage5_best_architecture": experiments[stage5_best_index]["architecture"],
        "saved_model": {"filename": "digits_net.joblib", **experiments[best_index],
                        "selection": "Highest observed test accuracy among six specified experiments; "
                                     "ties: fewer parameters, then listed order. No independent final holdout.",
                        "reload_predictions_identical": True},
        "test_indices": test_indices.tolist(),
        "baseline_test_predictions": prediction.tolist(),
    }
    (output / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Сохранена лучшая наблюдавшаяся модель:", tuple(experiments[best_index]["architecture"]),
          "test_accuracy=", experiments[best_index]["test_accuracy"], flush=True)
    print("Повторная загрузка joblib: предсказания совпали.", flush=True)
    print("Результаты:", output, flush=True)


if __name__ == "__main__":
    main()
