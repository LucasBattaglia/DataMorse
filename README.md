# ReadMe - Projet DataMorse


## Table des matières

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Utilisation](#3-utilisation)
4. [Fonctionnalités](#4-fonctionnalités)
5. [Contributeurs](#5-contributeurs)
6. [Licence](#6-licence)


## 1. Introduction

DataMorse est un protocole graphique innovant conçu pour encoder des messages sous forme de motifs de triangles, inspiré par le code Morse. Ce projet vise à offrir une méthode de transmission d'informations robuste et facilement identifiable, même en cas de déformation ou d'orientation aléatoire. L'implémentation est réalisée en Python, utilisant des bibliothèques graphiques pour faciliter la création et la lecture des données.

## 2. Installation

Pour exécuter le projet DataMorse, vous devez avoir Python installé sur votre machine. Suivez les étapes ci-dessous pour installer les dépendances nécessaires :

1. Clonez le dépôt :

```bash
git clone https://github.com/LucasBattaglia/DataMorse
cd DataMorse
```

2. Installez les bibliothèques requises :

```bash
pip install -r requirements.txt
```

## 3. Utilisation

Pour encoder un message en DataMorse, exécutez le script principal :

```bash
python3 DataMorse.py
```

Suivez-les instructions !
Le programme générera une matrice graphique représentant le message encodé. Vous pouvez également personnaliser l'arrière-plan en ajoutant un logo.

> > **Remarque :** _Le programme peut ne pas fonctionner avec tous les IDE, car il utilise OpenCV. Il est donc conseillé de l'exécuter directement dans le système de votre ordinateur (et non sous WSL, par exemple) et via la ligne de commande, plutôt que dans un IDE._

## 4. Fonctionnalités

**Encodage de données :** Transforme les lettres en motifs de triangles selon le code Morse.<br>
**Correction d'erreurs :** Intègre des algorithmes de Reed-Solomon et de Hamming pour assurer la fiabilité des données.<br>
**Masquage :** Applique des masques logiques pour éviter les zones blanches consécutives, facilitant ainsi la lecture.<br>
**Détection de position :** Utilise un motif de positionnement pour garantir l'orientation correcte des triangles.<br>

## 5. Contributeurs

* Lucas Battaglia - Développeur principal
* Chebbi Samar - Référent
* Patrice Laurancot - Responsable d’UE

## 6. Licence

Ce projet est sous licence. Pour plus de détails, veuillez consulter le fichier [LICENSE](LICENSE).


> **Contact :** Pour toute question ou suggestion, n'hésitez pas à contacter l'équipe de développement à l'adresse suivante : [lucas.battaglia@etu.uca.fr](mailto:lucas.battaglia@etu.uca.fr).

> Vous pouvez egalement obtenir plus de detail sur le projet dans le fichier [rapport.pdf](rapport.pdf)

**Merci d'utiliser DataMorse !**