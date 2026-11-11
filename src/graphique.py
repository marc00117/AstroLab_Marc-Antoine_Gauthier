"""
Interface graphique du simulateur (Levy). Seulement l'affichage : la logique se branche avec command=...

Documentation officielle :
    customtkinter : https://customtkinter.tomschimansky.com/documentation/
    matplotlib dans tkinter : https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html
    matplotlib (Axes) : https://matplotlib.org/stable/api/axes_api.html
    numpy : https://numpy.org/doc/stable/reference/
    placement grid : https://tkdocs.com/tutorial/grid.html
"""

import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import customtkinter as ctk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from src.astre import Astre, AstreFixe
from src.chargeur_csv import couleur_aleatoire, sauvegarder, os
from src.projectile import Projectile
from src.simulation import Simulation
from src.exceptions import DebutHistorique, FinHistorique

FOND = "#0f1424"   # bleu nuit derrière les astres


def texte(master, contenu: str, taille: int = 13, gras: bool = False) -> ctk.CTkLabel:
    """
    Crée une étiquette de texte.

    CTkLabel : https://customtkinter.tomschimansky.com/documentation/widgets/label
    CTkFont : https://customtkinter.tomschimansky.com/documentation/utility-classes/font

    :param master:
    :param contenu: Texte de l'étiquette
    :param taille: Taille de l'étiquette
    :param gras: Si le texte devrait être en gras, comme pour un titre
    :return: L'étiquette générée
    """
    return ctk.CTkLabel(master, text=contenu, font=ctk.CTkFont(size=taille, weight="bold" if gras else "normal"))


def bouton(master, contenu: str, commande=None, couleur=None) -> ctk.CTkButton:
    """
    Crée un bouton.

    CTkButton : https://customtkinter.tomschimansky.com/documentation/widgets/button

    :param master:
    :param contenu: Texte inscrit sur le bouton
    :param commande: Fonction appelée au clic (None = rien pour l'instant)
    :param couleur: Couleur du bouton
    :return: Le bouton généré
    """

    return ctk.CTkButton(master, text=contenu, command=commande, width=100, fg_color=couleur)


def ligne_de_widgets(widgets) -> None:
    """
    Place les widgets fournis côte à côte, de gauche à droite.

    :param widgets: Les widgets à mettre en ligne
    """
    # grid(row, column) range un widget dans une case d'une grille : https://tkdocs.com/tutorial/grid.html
    for colonne, widget in enumerate(widgets):
        widget.grid(row=0, column=colonne, padx=4, pady=8)


class BarreHaut(ctk.CTkFrame):
    """
    Barre du haut : Ouvrir, Enregistrer, Lancer, puis le fichier ouvert.

    CTkFrame (cadre qui regroupe des widgets) : https://customtkinter.tomschimansky.com/documentation/widgets/frame
    """

    def __init__(self, master, ouvrir=None, enregistrer=None, lancer=None, **kwargs):
        super().__init__(master, **kwargs)
        self.info = texte(self, "Aucun fichier")
        ligne_de_widgets([bouton(self, "Ouvrir un CSV…", ouvrir),
                          bouton(self, "Enregistrer…", enregistrer),
                          bouton(self, "Lancer un projectile…", lancer),
                          self.info])

    def afficher_fichier(self, nom: str, nb_astres: int, nb_projectiles: int):
        # configure change une option d'un widget déjà créé, ici son texte.
        self.info.configure(text=f"{nom} · {nb_astres} astres, {nb_projectiles} projectiles")


class GestionnaireObjets(ctk.CTkScrollableFrame):
    """
    Panneau de gauche : un tableau défilant avec une ligne par astre ou projectile.

    CTkScrollableFrame : https://customtkinter.tomschimansky.com/documentation/widgets/scrollableframe
    """

    COLONNES = ("Actif", "Nom", "Type", "Masse (MT)", "Distance (UA)", "Couleur")

    def __init__(self, master):
        super().__init__(master, width=480, label_text="Objets")
        self.afficher()

    def afficher(self, astres=(), projectiles=()):
        """
        Efface le tableau puis le réécrit.

        :param astres:
        :param projectiles:
        :return:
        """

        # winfo_children = widgets contenus, destroy = les supprimer
        for widget in self.winfo_children():
            widget.destroy()

        for colonne, titre in enumerate(self.COLONNES):
            texte(self, titre, gras=True).grid(row=0, column=colonne, padx=6, sticky="w")

        for ligne, objet in enumerate(astres + projectiles, start=1):
            est_astre = objet in astres

            # CTkCheckBox (select = cocher) : https://customtkinter.tomschimansky.com/documentation/widgets/checkbox
            case = ctk.CTkCheckBox(self, text="", width=30, command=self.CommandeCase(objet))
            if objet.actif:
                case.select()
            case.grid(row=ligne, column=0)

            # np.linalg.norm : longueur du vecteur (x, y), donc distance au centre
            # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            distance = objet.distance if est_astre else np.linalg.norm(objet.pos0)
            valeurs = (objet.nom, "astre" if est_astre else "projectile", f"{objet.masse:.3g}", f"{distance:.2f}")
            for colonne, valeur in enumerate(valeurs, start=1):
                texte(self, valeur).grid(row=ligne, column=colonne, padx=6, sticky="w")
            # Pastille de couleur : une étiquette vide dont fg_color est la couleur de fond
            ctk.CTkLabel(self, text="", width=30, height=16, fg_color=objet.couleur, corner_radius=4
                         ).grid(row=ligne, column=5)

    class CommandeCase:
        """
        Une classe-usine qui génère une fonction qui alterne l'état d'activité d'un Astre,
        mais demande que le graphique soit recalculé aussi. Utilisable comme commande pour un widget CTk.

        *"Toute opportunité d'utiliser une classe-usine est une bonne opportunité." - Moi, 2026*
        """
        def __init__(self, objet: Astre):
            self.objet: Astre = objet

        def __call__(self):
            if self.objet.actif:
                self.objet.actif = False
            else:
                self.objet.actif = True


class VueGraphique(ctk.CTkFrame):
    """
    Panneau du centre : le graphique matplotlib dans la fenêtre, avec la barre zoom/déplacement.
    """

    def __init__(self, master):
        super().__init__(master)

        # Figure = la zone de dessin, add_subplot(111) = un seul graphique (Axes) dedans
        # https://matplotlib.org/stable/api/figure_api.html
        self.fig = plt.Figure(figsize=(6, 6), facecolor=FOND)
        self.ax = self.fig.add_subplot(111)
        # Marges fixes pour les titres des axes, réglées une seule fois (tight_layout à chaque image était lent)
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.subplots_adjust.html
        self.fig.subplots_adjust(left=0.1, right=0.97, top=0.94, bottom=0.08)

        # FigureCanvasTkAgg met la figure dans tkinter, NavigationToolbar2Tk ajoute les boutons zoom/déplacement
        # https://matplotlib.org/stable/api/backend_tk_api.html
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False).pack(side="bottom", fill="x")

        # pack empile les widgets ; fill="both" et expand=True : le graphique prend toute la place
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.dessiner()

    def dessiner(self, astres=(), projectiles=(), t: float = 0.0):
        """
        Dessine au temps t les orbites et les objets actifs avec leur nom.

        :param astres: Les AstreFixe à afficher
        :param projectiles: Les Projectiles à afficher
        :param t: Le moment auquel on souhaite illustrer le tout.
        """

        ax = self.ax
        ax.clear()

        actifs = [objet for objet in list(astres) + list(projectiles) if objet.actif]

        # Vue carrée qui contient l'astre le plus loin (+10 %)
        limite = 1.1 * max([np.linalg.norm(objet.position(t)) for objet in actifs if isinstance(objet, AstreFixe)] + [1])

        # Axes.set : https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html
        ax.set(xlim=(-limite, limite), ylim=(-limite, limite), aspect="equal", facecolor=FOND,
               title=f"t = {t} jours", xlabel="x (UA)", ylabel="y (UA)")

        # Texte en gris clair, sinon il est noir sur le fond bleu nuit
        # tick_params : https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
        ax.tick_params(colors="lightgray")
        for etiquette in (ax.title, ax.xaxis.label, ax.yaxis.label):
            etiquette.set_color("lightgray")

        # Trajectoire complète des projectiles déjà calculés
        # historique[:, 0] = tous les x, historique[:, 1] = tous les y : https://numpy.org/doc/stable/user/basics.indexing.html
        # plot relie les points par une ligne : https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
        for projectile in projectiles:
            if projectile.actif and len(projectile.historique) > 1:
                ax.plot(projectile.historique[:, 0], projectile.historique[:, 1], color=projectile.couleur,
                        linewidth=0.8, alpha=0.7)

        for objet in actifs:
            # np.sqrt : racine carrée, pour que le Soleil (taille 109) ne cache pas tout
            taille = max(8, 20 * np.sqrt(objet.taille)) if objet in astres else 25

            if objet in astres:
                x, y = objet.position(t)

                if objet.distance > 0:
                    # Circle + add_patch : cercle de l'orbite autour du Soleil ou du parent
                    # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Circle.html
                    centre = objet.parent.position(t) if objet.parent else (0, 0)
                    ax.add_patch(plt.Circle(centre, objet.distance, fill=False, color=objet.couleur, alpha=0.4))

                # np.sqrt : racine carrée, pour que le Soleil (taille 109) ne cache pas tout
                taille = max(8, 20 * np.sqrt(objet.taille)) if objet in astres else 25

                # scatter dessine un point, annotate écrit un texte à côté
                ax.scatter(x, y, taille, objet.couleur, zorder=3)
                ax.annotate(objet.nom, (x, y), xytext=(4, 2), textcoords="offset points", color="white", fontsize=7)

            elif objet in projectiles:
                try:
                    x, y = objet.position(t)
                    ax.scatter(x, y, taille, objet.couleur, zorder=3)
                    ax.annotate(objet.nom, (x, y), xytext=(4, 2), textcoords="offset points", color="white", fontsize=7)
                except DebutHistorique:
                    x, y = objet.pos0
                    ax.scatter(x, y, taille, objet.couleur, zorder=3, marker="*")
                    ax.annotate(objet.nom, (x, y), xytext=(4, 2), textcoords="offset points", color="white", fontsize=7)
                except FinHistorique:
                    match objet.etat:
                        case "expire":
                            x, y = objet.historique[-1, 0], objet.historique[-1, 1]
                        case "hors systeme":
                            x, y = objet.historique[-1, 0], objet.historique[-1, 1]
                        case "collision":
                            x, y = objet.cible.position(t)
                        case _:
                            raise ValueError("État du projectile invalide!")
                    ax.scatter(x, y, taille, objet.couleur, zorder=3, marker="x")
                    ax.annotate(objet.nom, (x, y), xytext=(4, 2), textcoords="offset points", color="white", fontsize=7)

        self.canvas.draw_idle()  # redessine la figure dans la fenêtre dès que possible


class DetailsProjectile(ctk.CTkFrame):
    """
    Panneau de droite : choix d'un projectile, ses informations et le bouton d'export.
    """

    INFOS = ("État", "Cible", "Position (UA)", "Vitesse (UA/j)", "Masse (MT)")

    def __init__(self, master, exporter=None):
        super().__init__(master)

        texte(self, "Projectile", 18, True).grid(row=0, column=0, columnspan=2, padx=12, pady=8, sticky="w")

        # CTkComboBox (liste déroulante) : https://customtkinter.tomschimansky.com/documentation/widgets/combobox
        self.choix = ctk.CTkComboBox(self, values=["Aucun projectile"], state="readonly")
        self.choix.grid(row=1, column=0, columnspan=2, padx=12, sticky="ew")
        self.valeurs = {}  # libellé -> étiquette où écrire la valeur
        for ligne, nom in enumerate(self.INFOS, start=2):
            texte(self, nom).grid(row=ligne, column=0, padx=12, pady=3, sticky="w")
            self.valeurs[nom] = texte(self, "—")
            self.valeurs[nom].grid(row=ligne, column=1, padx=12, sticky="w")

        bouton(self, "Exporter la trajectoire…", exporter
               ).grid(row=len(self.INFOS) + 2, column=0, columnspan=2, padx=12, pady=12)

    def afficher(self, noms=(), infos=None):
        # Remplit la liste des projectiles et les valeurs (« — » si une valeur manque).
        noms = list(noms) or ["Aucun projectile"]
        self.choix.configure(values=noms)
        self.choix.set(noms[0])  # set choisit l'élément affiché
        for nom, etiquette in self.valeurs.items():
            etiquette.configure(text=str((infos or {}).get(nom, "—")))


class BarreControle(ctk.CTkFrame):
    """
    Barre du bas : réinitialiser, reculer, jouer, arrêter, avancer, le pas dt et le temps actuel.
    """

    def __init__(self, master):
        super().__init__(master)
        self.master: FenetreSimulateur

        # CTkEntry (case où l'on écrit) : https://customtkinter.tomschimansky.com/documentation/widgets/entry
        self.pas_temps = ctk.CTkEntry(self, width=60)
        self.pas_temps.insert(0, "5")  # texte de départ
        self.pas_temps.bind("<FocusIn>", self.__commande_pause)

        # Défilement du temps
        self.defil = False

        # Variable de temps
        self.__t: int = 0

        # Boutons rangés par nom pour brancher la logique plus tard, ex. barre.boutons["jouer"].configure(command=...)
        self.boutons = {
            "reinitialiser": bouton(self, "Réinitialiser", commande=self.__commande_reinitialiser),
            "reculer": bouton(self, "◀ Reculer", commande=self.__commande_reculer),
            "jouer": bouton(self, "▶ Jouer", couleur="#2e8b57", commande=self.__commande_lecture),
            "arreter": bouton(self, "■ Arrêter", couleur="#c0392b", commande=self.__commande_arreter),
            "avancer": bouton(self, "Avancer ▶", commande=self.__commande_avancer)
        }
        self.temps = texte(self, "", gras=True)
        ligne_de_widgets(list(self.boutons.values()) + [texte(self, "pas de temps (jours)"), self.pas_temps, self.temps])
        self.afficher_temps(0)

    def afficher_temps(self, t: int):
        self.temps.configure(text=f"t = {t} jours")

    @property
    def t(self):
        return self.__t

    @t.setter
    def t(self, value):
        self.__t = max(0, min(value, int(self.master.simulateur.temps_max())))  # Garder t entre 0 et temps_max
        self.afficher_temps(self.__t)
        self.master.simulateur.calculer(value)
        self.master.vue.dessiner(*self.master.simulateur.astres_et_projectiles, self.t)

    # Commandes des boutons
    def __commande_reinitialiser(self):
        """Réinitialise le temps et force le graphique à se recalculer."""
        self.__commande_arreter()
        self.master.simulateur.calculer(self.t)
        self.master.afficher(avec_cache=True)
        

    def __commande_arreter(self):
        """Arrête le défilement du temps et revient à t=0."""
        self.t = 0
        self.__commande_pause()

    def __commande_lecture(self):
        """Fait défiler le temps."""
        if not self.defil and self.t < self.master.simulateur.temps_max():
            self.defil = True
            self.mise_a_jour_lecture()
            self.boutons["jouer"].configure(text="⏸ Pause", command=self.__commande_pause)
        else:
            self.__commande_pause()

    def __commande_pause(self, evenement = None): # evenement permet d'etre utiliser avec pas_temps
        """Met sur pause le défilement du temps."""
        self.defil = False
        self.boutons["jouer"].configure(text="▶ Jouer", command=self.__commande_lecture)

    def __commande_avancer(self):
        """Avance de dt."""
        self.t += int(self.pas_temps.get())

    def __commande_reculer(self):
        """Recule de dt."""
        self.t -= int(self.pas_temps.get())

    def mise_a_jour_lecture(self):
        """Met à jour le temps à chaque 0.01 seconde"""
        
        #if self.pas_temps._is_focused: # si l'on change le temps en emem temps que sa joue
        #    self.__commande_pause()
        
        print("1- ", self.focus_get(),"    2- ",  self.boutons["jouer"].focus_get()," rep = ", self.focus_get() != self.boutons["jouer"].focus_get())
        
        
        
        if not str(self.focus_get()).startswith(str(self.boutons["jouer"])): ### mheeee je oense pas que ca je vais aimer ca:( 
            self.__commande_pause()
        
        if self.defil and self.t < self.master.simulateur.temps_max():
            if self.pas_temps.get() != '':
                self.t += int(self.pas_temps.get())
            self.after(10, self.mise_a_jour_lecture)  # programmé avant de dessiner : l'attente et le dessin se chevauchent
            
        else:
            self.__commande_pause()


class FormulaireProjectile(ctk.CTkToplevel):
    """
    Fenêtre « Lancer un projectile » : on saisit la position et la vitesse, le projectile part au temps actuel.
    La couleur est choisie automatiquement.

    CTkToplevel (fenêtre secondaire) : https://customtkinter.tomschimansky.com/documentation/windows/toplevel
    """

    # (clé, libellé, valeur proposée) ; repère : vitesse circulaire à 1 UA ≈ 0,0172 UA/j
    CHAMPS = (("nom", "Nom", "Projectile"), ("x0", "x (UA)", "1.5"), ("y0", "y (UA)", "0"),
              ("vx0", "vx (UA/j)", "0"), ("vy0", "vy (UA/j)", "0.014"), ("masse", "Masse (MT)", "0.001"),
              ("rayon", "Rayon (UA)", "0.000001"), ("duree_vie", "Durée de vie (jours)", "1000"))

    def __init__(self, master, lancer):
        super().__init__(master)
        self.title("Lancer un projectile")
        self.lancer = lancer  # fonction qui reçoit le nom et les nombres, et renvoie un message d'erreur ou None
        self.entrees = {}
        for ligne, (cle, libelle, defaut) in enumerate(self.CHAMPS):
            texte(self, libelle).grid(row=ligne, column=0, padx=12, pady=3, sticky="w")
            self.entrees[cle] = ctk.CTkEntry(self, width=160)
            self.entrees[cle].insert(0, defaut)
            self.entrees[cle].grid(row=ligne, column=1, padx=12, pady=3)
        self.erreur = ctk.CTkLabel(self, text="", text_color="#FF6B6B", wraplength=300)
        self.erreur.grid(row=len(self.CHAMPS), column=0, columnspan=2, padx=12)
        bouton(self, "Lancer", self.valider).grid(row=len(self.CHAMPS) + 1, column=0, columnspan=2, pady=10)
        self.after(100, self.grab_set)  # garde clavier et souris sur le formulaire une fois affiché

    def valider(self):
        """Vérifie les champs ; ferme la fenêtre si le projectile a été lancé, sinon affiche l'erreur."""
        nom = self.entrees["nom"].get().strip()
        try:
            # replace : accepte aussi la virgule décimale (1,5)
            nombres = {cle: float(entree.get().replace(",", ".")) for cle, entree in self.entrees.items() if cle != "nom"}
        except ValueError:
            nombres = None
        if not nom:
            erreur = "Le nom est obligatoire."
        elif nombres is None:
            erreur = "Position, vitesse, masse, rayon et durée de vie doivent être des nombres."
        elif nombres["duree_vie"] <= 0:
            erreur = "La durée de vie doit être strictement positive."
        else:
            erreur = self.lancer(nom, nombres)
        if erreur:
            self.erreur.configure(text=erreur)
        else:
            self.destroy()


class FenetreSimulateur(ctk.CTk):
    """
    Fenêtre complète : barre du haut, objets à gauche, graphique au centre, projectile à droite, contrôles en bas.
    CTk (fenêtre principale) : https://customtkinter.tomschimansky.com/documentation/windows/window
    """

    def __init__(self):
        super().__init__()
        self.title("AstroLab : Simulateur de projectiles dans l'espace")
        self.geometry("1400x800")
        self.protocol("WM_DELETE_WINDOW", self.quit)  # NE PAS ENLEVER : ferme proprement la fenêtre

        self.simulateur = Simulation()
        self.barre_haut = BarreHaut(self, ouvrir=self.ouvrir_csv, enregistrer=self.enregistrer_csv,
                                    lancer=lambda: FormulaireProjectile(self, self.lancer_projectile))
        self.gestionnaire = GestionnaireObjets(self)
        self.vue = VueGraphique(self)
        self.details = DetailsProjectile(self, exporter=self.exporter_trajectoire)
        self.barre_controle = BarreControle(self)

        # sticky="nsew" : le panneau s'étire dans sa case ; columnspan=3 : la barre prend les 3 colonnes
        self.barre_haut.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=4)
        for colonne, panneau in enumerate((self.gestionnaire, self.vue, self.details)):
            panneau.grid(row=1, column=colonne, sticky="nsew", padx=4, pady=4)
        self.barre_controle.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=4)

        # weight=1 : la ligne 1 et la colonne 1 (le graphique) grandissent avec la fenêtre
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.cache = {
            "astres": [],
            "projectiles": [],
            "nom_fichier": "",
            "t": 0.0
        }

    def afficher(self, astres=(), projectiles=(), nom_fichier: str = "Aucun fichier", t: float = 0.0,
                 avec_cache: bool = False):
        """
        Remplit tous les panneaux avec les objets donnés.

        Régénère également la cache avec les paramètres fournis.

        :param avec_cache: Si True, rappelle la fonction en utilisant les paramètres en cache. À éviter si possible.
        :param astres: Les astres fixes à afficher
        :param projectiles: Les projectiles à afficher
        :param nom_fichier: Nom du fichier
        :param t: Le temps courant
        """
        if avec_cache:
            self.afficher(
                astres=self.cache["astres"],
                projectiles=self.cache["projectiles"],
                nom_fichier=self.cache["nom_fichier"],
                t=self.cache["t"])
        else:
            self.barre_haut.afficher_fichier(nom_fichier, len(astres), len(projectiles))
            self.gestionnaire.afficher(astres, projectiles)
            self.vue.dessiner(astres, projectiles, t)
            infos = {"État": projectiles[0].etat, "Masse (MT)": f"{projectiles[0].masse:g}"} if projectiles else None
            self.details.afficher([projectile.nom for projectile in projectiles], infos)
            self.barre_controle.afficher_temps(t)

            # Cache
            self.cache["astres"] = astres
            self.cache["projectiles"] = projectiles
            self.cache["nom_fichier"] = nom_fichier
            self.cache["t"] = t

    def charger(self, chemin: str):
        """
        Charge un CSV dans la simulation, calcule les trajectoires puis affiche le tout à t = 0.

        :param chemin: Chemin du fichier CSV
        """
        self.simulateur.importer(chemin)
        self.simulateur.calculer(self.barre_controle.t)
        
        # os.path.basename garde seulement le nom du fichier : https://docs.python.org/3/library/os.path.html
        self.afficher(*self.simulateur.astres_et_projectiles, os.path.basename(chemin))
        self.barre_controle.t = 0

    def ouvrir_csv(self):
        """Demande un fichier CSV et le charge dans la simulation."""
        # askopenfilename renvoie le chemin choisi ("" si annulé)
        # https://docs.python.org/3/library/dialog.html#tkinter.filedialog.askopenfilename
        chemin = filedialog.askopenfilename(filetypes=[("Fichiers CSV", "*.csv")], parent=self)
        if not chemin:
            return
        try:
            self.charger(chemin)
        except (OSError, KeyError, ValueError) as erreur:  # OSError couvre FileNotFoundError et PermissionError
            messagebox.showerror("Le chargement a échoué !", str(erreur), parent=self)

    def enregistrer_csv(self):
        """Demande où enregistrer, puis écrit les astres et projectiles dans un CSV rechargeable."""
        # asksaveasfilename : defaultextension ajoute .csv si oublié
        # https://docs.python.org/3/library/dialog.html#tkinter.filedialog.asksaveasfilename
        chemin = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Fichiers CSV", "*.csv")], parent=self)
        if not chemin:
            return
        try:
            sauvegarder(self.simulateur, chemin)
        except (OSError, ValueError) as erreur:
            messagebox.showerror("L'enregistrement a échoué !", str(erreur), parent=self)

    def lancer_projectile(self, nom: str, n: dict):
        """
        Lance au temps actuel le projectile saisi dans le formulaire, avec une couleur choisie automatiquement.

        :param nom: Nom du projectile
        :param n: Nombres saisis (x0, y0, vx0, vy0, masse, rayon, duree_vie)
        :return: Un message d'erreur, ou None si le projectile a été lancé
        """
        if not self.simulateur.astres:
            return "Ouvrez d'abord un CSV qui contient des astres."
        objets = self.simulateur.astres + self.simulateur.projectiles
        if nom in [objet.nom for objet in objets]:
            return f"Le nom « {nom} » est déjà utilisé."
        couleur = couleur_aleatoire({objet.couleur for objet in objets})  # différente des couleurs déjà prises
        try:
            projectile = Projectile(nom, n["masse"], couleur, 0.5, n["rayon"], (n["x0"], n["y0"]), (n["vx0"], n["vy0"]),
                                    self.barre_controle.t, n["duree_vie"])
        except ValueError as erreur:  # masse ou rayon négatifs
            return str(erreur)
        self.simulateur.projectiles.append(projectile)
        self.simulateur.calculer(self.barre_controle.t)
        self.afficher(*self.simulateur.astres_et_projectiles, self.cache["nom_fichier"], self.barre_controle.t)
        return None

    def exporter_trajectoire(self):
        """Écrit la trajectoire du projectile choisi dans un CSV : une ligne par jour (t, x, y, vx, vy)."""
        nom = self.details.choix.get()
        projectile = next((p for p in self.simulateur.projectiles if p.nom == nom), None)
        if projectile is None or len(projectile.historique) < 2:
            messagebox.showinfo("Exporter la trajectoire", "Aucune trajectoire calculée pour ce projectile.", parent=self)
            return
        chemin = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Fichiers CSV", "*.csv")],
                                              initialfile=f"trajectoire_{nom}.csv", parent=self)
        if not chemin:
            return
        # Une ligne de l'historique par jour depuis l'apparition, d'où t = temps_apparition + numéro de ligne
        # np.arange(n) donne 0, 1, ..., n-1 : https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        trajectoire = pd.DataFrame(projectile.historique, columns=["x", "y", "vx", "vy"])
        trajectoire.insert(0, "t", projectile.temps_apparition + np.arange(len(trajectoire)))
        try:
            trajectoire.to_csv(chemin, index=False)
        except OSError as erreur:
            messagebox.showerror("L'export a échoué !", str(erreur), parent=self)


# Aperçu de l'interface : « python -m src.graphique » depuis la racine du projet
if __name__ == "__main__":
    fenetre = FenetreSimulateur()
    fenetre.charger("donnees/systeme_solaire.csv")
    fenetre.mainloop()
