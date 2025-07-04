from PIL import Image
import numpy as np

def clean_logo(image_path, output_path, tolerance=40):
    # Charger l'image en RGBA
    img = Image.open(image_path).convert("RGBA")
    img_np = np.array(img)

    # Prendre la couleur du fond (coin haut gauche)
    bg_color = img_np[0, 0, :3]

    # Créer un masque : pixels proches du fond → True
    diff = np.abs(img_np[:, :, :3] - bg_color)
    mask_bg = np.all(diff < tolerance, axis=-1)

    # Appliquer le masque : fond devient transparent
    img_np[mask_bg] = [0, 0, 0, 0]

    # Recadrer autour du contenu utile (où alpha > 0)
    alpha = img_np[:, :, 3]
    non_empty_rows = np.where(np.any(alpha > 0, axis=1))[0]
    non_empty_cols = np.where(np.any(alpha > 0, axis=0))[0]

    if non_empty_rows.size and non_empty_cols.size:
        top, bottom = non_empty_rows[0], non_empty_rows[-1] + 1
        left, right = non_empty_cols[0], non_empty_cols[-1] + 1
        img_np_cropped = img_np[top:bottom, left:right]
    else:
        img_np_cropped = img_np  # Rien à recadrer

    # Sauvegarder l'image nettoyée et recadrée
    Image.fromarray(img_np_cropped).save(output_path)

# Exemple d’utilisation
clean_logo(
    'App/static/images/logo_promptune.png',
    'App/static/images/logo_promptune_clean.png',
    tolerance=40  # Ajuste si ton fond est très proche du logo
)
