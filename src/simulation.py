from src.astre import AstreFixe
from src.chargeur_csv import charger_csv
from src.projectile import Projectile
from src.physique import calculer_trajectoire
from src.exceptions import *


class Simulation:
    """Contient la liste des astres et projectiles, et génère leurs trajectoires via physique_Kepler"""
    def __init__(self):
        self.astres: list[AstreFixe] = []
        self.projectiles: list[Projectile] = []
        self.dt = 0.1

    @property
    def astres_et_projectiles(self):
        return self.astres, self.projectiles

    def temps_max(self, padt=50):
        """
        :param padt: Ajoute un peu de temps après la disparition du dernier projectile.
        :return: Le temps maximal de la ligne du temps
        """
        t_max = 0
        if self.projectiles:
            for projectile in self.projectiles:
                temps_mort = projectile.temps_apparition + len(projectile.historique)
                if temps_mort > t_max:
                    t_max = temps_mort
        return t_max + padt

    def calculer(self, t):
        #prendre en compte le temps apparition!!!
        for projectile in self.projectiles:
            if projectile.actif:
            #    projectile.reinitialiser()  # Efface l'ancien historique avant de recalculer
                try:
                    projectile.cible = calculer_trajectoire(projectile, self.astres, t, self.dt,
                                                            int(projectile.temps_apparition + projectile.duree_vie))
                    if projectile.cible is not None:
                        projectile.etat = "collision"
                except TempsMaximumAtteinte:
                    projectile.etat = "expire"
                except HorsSystemeError:
                    print(f"Le projectile {projectile.nom} est sorti du système!"
                          f"Portez attention aux valeurs de vit0_x et vit0_y, elles sont en UA/jour.")
                    projectile.etat = "hors systeme"

    def importer(self, fichier):
        astres, projectiles = charger_csv(fichier)
        self.astres = astres.copy()
        self.projectiles = projectiles.copy()
