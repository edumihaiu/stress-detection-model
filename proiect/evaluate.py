# evaluate_model.py
# Rulează după antrenare pentru a genera matricea de confuzie și raportul complet
# Folosire: python evaluate_model.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import config
from dataset import load_datasets

# ─── Setări ───────────────────────────────────────────────────────────────────
CLASS_NAMES = ['😊 Relaxat (0)', '😐 Neutru (1)', '😰 Stresat (2)']
CLASS_NAMES_SHORT = ['Relaxat', 'Neutru', 'Stresat']
MODEL_PATH = "../best_model.keras"   # schimbă dacă e altundeva
OUTPUT_DIR = "./evaluare_rezultate"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Încarcă modelul și datele ────────────────────────────────────────────────
def load_model_and_predict():
    print("[*] Încarc modelul...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("[*] Încarc datele de test...")
    _, test_ds = load_datasets()

    print("[*] Generez predicții...")
    y_true = []
    y_pred = []
    y_prob = []  # probabilitățile brute (pentru analiza confidenței)

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ─── Matricea de confuzie ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f1a')

    cmap = LinearSegmentedColormap.from_list(
        'stress_cmap', ['#0f0f1a', '#1a1a3e', '#2d2d7a', '#4a4abf', '#6b6bff', '#a78bfa']
    )

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_percent],
        ['Număr absolut de predicții', 'Procentaj per clasă reală (%)'],
        ['d', '.1f']
    ):
        ax.set_facecolor('#0f0f1a')
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=CLASS_NAMES_SHORT,
            yticklabels=CLASS_NAMES_SHORT,
            ax=ax,
            linewidths=0.5,
            linecolor='#2a2a4a',
            annot_kws={"size": 14, "weight": "bold", "color": "white"},
            cbar_kws={'shrink': 0.8}
        )
        ax.set_title(title, color='#a78bfa', fontsize=13, pad=15, fontweight='bold')
        ax.set_xlabel('Predicție model', color='#8888cc', fontsize=11)
        ax.set_ylabel('Clasă reală', color='#8888cc', fontsize=11)
        ax.tick_params(colors='#8888cc', labelsize=10)

        # Colorează bara de culoare
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='#8888cc')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8888cc')

    plt.suptitle(
        'Matrice de Confuzie — Model Detecție Stres',
        color='white', fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"[✓] Salvat: {path}")
    return cm


# ─── Analiza erorilor pe clase ────────────────────────────────────────────────
def plot_per_class_analysis(y_true, y_pred, y_prob):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0f1a')

    colors = ['#4ade80', '#60a5fa', '#f87171']
    
    for cls in range(3):
        ax = axes[cls]
        ax.set_facecolor('#161628')
        
        # Probabilitățile pentru această clasă, împărțite pe predicții corecte/greșite
        mask_real = (y_true == cls)
        probs_cls = y_prob[mask_real, cls]
        pred_cls = y_pred[mask_real]
        
        correct_probs = probs_cls[pred_cls == cls]
        wrong_probs = probs_cls[pred_cls != cls]

        bins = np.linspace(0, 1, 20)
        ax.hist(correct_probs, bins=bins, alpha=0.8, color=colors[cls],
                label=f'Corect ({len(correct_probs)})', edgecolor='none')
        ax.hist(wrong_probs, bins=bins, alpha=0.5, color='#ef4444',
                label=f'Greșit ({len(wrong_probs)})', edgecolor='none')

        ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.4, linewidth=1)
        
        acc = len(correct_probs) / len(probs_cls) * 100 if len(probs_cls) > 0 else 0
        ax.set_title(
            f'{CLASS_NAMES_SHORT[cls]}\nAcuratețe: {acc:.1f}%',
            color='white', fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('Probabilitate prezisă', color='#8888cc', fontsize=10)
        ax.set_ylabel('Număr imagini', color='#8888cc', fontsize=10)
        ax.tick_params(colors='#8888cc')
        ax.legend(fontsize=9, facecolor='#0f0f1a', labelcolor='white')
        ax.spines['bottom'].set_color('#2a2a4a')
        ax.spines['left'].set_color('#2a2a4a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(
        'Distribuția Probabilităților per Clasă',
        color='white', fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'per_class_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"[✓] Salvat: {path}")


# ─── Analiza false positives ──────────────────────────────────────────────────
def analyze_false_positives(y_true, y_pred, y_prob):
    print("\n" + "═" * 60)
    print("  ANALIZA FALSE POSITIVES")
    print("═" * 60)

    for cls in range(3):
        # False positives = prezis ca cls, dar e altceva
        fp_mask = (y_pred == cls) & (y_true != cls)
        fp_count = fp_mask.sum()
        total_pred_as_cls = (y_pred == cls).sum()

        print(f"\n  Clasa '{CLASS_NAMES_SHORT[cls]}':")
        print(f"    False Positives: {fp_count} / {total_pred_as_cls} preziceri")

        if fp_count > 0:
            fp_probs = y_prob[fp_mask, cls]
            fp_true = y_true[fp_mask]
            print(f"    Confidență medie FP: {fp_probs.mean():.3f}")
            print(f"    Confidență max FP:   {fp_probs.max():.3f}")
            print(f"    Vin din clasele reale:")
            for src_cls in range(3):
                if src_cls != cls:
                    count = (fp_true == src_cls).sum()
                    if count > 0:
                        print(f"      → '{CLASS_NAMES_SHORT[src_cls]}': {count} cazuri")

    print("\n" + "═" * 60)


# ─── Raport text complet ──────────────────────────────────────────────────────
def print_full_report(y_true, y_pred):
    print("\n" + "═" * 60)
    print("  RAPORT COMPLET CLASIFICARE")
    print("═" * 60)
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES_SHORT,
        digits=4
    )
    print(report)

    # Salvează și în fișier
    path = os.path.join(OUTPUT_DIR, 'raport_clasificare.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("RAPORT COMPLET CLASIFICARE\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"[✓] Raport salvat: {path}")


# ─── Sugestii automate bazate pe rezultate ────────────────────────────────────
def print_suggestions(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    print("\n" + "═" * 60)
    print("  SUGESTII AUTOMATE")
    print("═" * 60)

    # Verifică dacă "stresat" e over-prezis
    stressed_fp = ((y_pred == 2) & (y_true != 2)).sum()
    stressed_total = (y_pred == 2).sum()
    if stressed_total > 0 and stressed_fp / stressed_total > 0.3:
        print("\n  ⚠️  'Stresat' e over-prezis (>30% false positives)")
        print("     → Încearcă prag mai mare: pred[2] > 0.65 în loc de 0.5")
        print("     → Sau echilibrează clasele prin undersampling")

    # Verifică confuzie neutru↔stresat
    if cm_percent[1][2] > 20:
        print(f"\n  ⚠️  {cm_percent[1][2]:.1f}% din 'Neutru' e clasificat ca 'Stresat'")
        print("     → Scoate 'surprise' din clasa Neutru din dataset")
        print("     → Sau adaugă mai multe exemple de neutru clar")

    # Verifică confidența medie
    mean_conf = np.max(y_prob, axis=1).mean()
    if mean_conf > 0.9:
        print(f"\n  ⚠️  Confidență medie foarte mare ({mean_conf:.3f}) → posibil overfit")
        print("     → Verifică dacă val_accuracy < train_accuracy în history")
    elif mean_conf < 0.6:
        print(f"\n  ⚠️  Confidență medie mică ({mean_conf:.3f}) → model nesigur")
        print("     → Încearcă mai multe epoci sau augmentare mai agresivă")

    overall_acc = (y_true == y_pred).mean() * 100
    print(f"\n  📊 Acuratețe generală: {overall_acc:.2f}%")
    print("═" * 60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  EVALUARE MODEL — DETECȚIE STRES FACIAL")
    print("═" * 60 + "\n")

    y_true, y_pred, y_prob = load_model_and_predict()

    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_analysis(y_true, y_pred, y_prob)
    print_full_report(y_true, y_pred)
    analyze_false_positives(y_true, y_pred, y_prob)
    print_suggestions(y_true, y_pred, y_prob)

    print(f"\n[✓] Toate rezultatele salvate în: {OUTPUT_DIR}/")
    print("    → confusion_matrix.png")
    print("    → per_class_analysis.png")
    print("    → raport_clasificare.txt\n")