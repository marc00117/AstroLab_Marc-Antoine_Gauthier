from src.projectile import Projectile
from src.astre import AstreFixe, G
from src.exceptions import HorsSystemeError, TempsMaximumAtteinte

import numpy as np

DISTANCE_MAX_AJOUTER = 2  # Distance en UA ajoutées pour les bornes maximales du projectile pour le considérer comme 'hors du système'


def calculer_trajectoire(projectile: Projectile, liste_astre: list[AstreFixe], prochain_t: int, dt: float = 0.1, temps_maximum: int = 3653) -> AstreFixe:
    
    """_Crée la trajectoire du projectile en calculant sa position et sa vitesse sous les gravités des astres du système.
    Les états calculés jusqu'au prochain temps sont enregistrés dans l'historique du projectile, s'il nest pas en se moment dans historique.
    Retourne l'astre fixe toucher ou Souleve un Exception selon ce qui ce passe à la place_

    Args:
        projectile (Projectile): _Projectille utiliser pour la trajectoire_
        liste_astre (list[AstreFixe]): _Système des astres fixes dans une liste_
        prochain_t (int) : _Prochain temps qui viens_
        dt (float, optional): _pas de temps entre les positions (en jour)_. Initialiser à 0.1 (recommandé pour la rapidité) si plus précis et un peux moins rapide -> 0.01.
        temps_maximum (int, optional): _nombre de position calculée maximum_. Initialisé à 3653 jours (10 ans).

    Raises:
        HorsSystemeError: _si le projectile est hors du système._
        TempsMaximumAtteinte: _si le nombre de temps maximum a été atteint._

    Returns:
        AstreFixe: _retourne l'astre fixe seulement s'il a eu contacte avec un astre fixe_
    """
    historique = projectile.historique
    if len(historique) > prochain_t: # verifi si la trajectoire au temps prochain_t est deja calculee 
        return 
    
    # Vérifi si toutes les valeurs données sont correct
    if not isinstance(projectile, Projectile):
        raise ValueError("Physique : p doit etre un Projectile.")
    if not all(isinstance(astre, AstreFixe) for astre in liste_astre):
        raise ValueError(
            "Physique : Il faut que toutes les astres dans liste_astre soit des AstreFixe."
        )
    if not (isinstance(prochain_t, (int)) and isinstance(temps_maximum, int)):
        raise ValueError("Physique : prochain_t et temps_maximum doivent être des entiers.")
    if not isinstance(dt, (int, float)):
        raise ValueError("Physique : dt doit être un nombre.")
    if prochain_t < 0 or dt < 0 or temps_maximum < 0:
        raise ValueError("Physique : prochain_t, dt et temps_maximum doivent être positives.")
    
    historique = projectile.historique
    
    if len(historique) > prochain_t: # verifi si la trajectoire au temps prochain_t est deja calculee 
        return 

    Nb_iterations_enregistrement = int(1 / dt) # calcule qui determine le nombre iteration pour une journee

    # détermine la longueur maximale pour arrêter la création de la liste
    astre_loin = max(liste_astre, key=lambda astre: astre.distance)
    distance_max = astre_loin.distance + DISTANCE_MAX_AJOUTER

    etat_au_temps = historique[-1]
    
    # Inisialise les variables utilisers
    pos, vit = np.array([float(etat_au_temps[0]), float(etat_au_temps[1])]), np.array([float(etat_au_temps[2]), float(etat_au_temps[3])])    
    t = projectile.temps_apparition + len(historique) - 1   # en jour : les astres partent de leur position au lancement
    nombre_diteration_max = int(temps_maximum / dt) # nombre ditteration que la boucle peux faire pour le programme

    for i in range(int(len(historique)/dt), int((prochain_t + 1)/dt)):#'''1 = buffer pour que ca marche''' 
        
        # Enregistre les coordonnées des astres ici, ce qui nous évite de les chercher constamment.
        astres_positions = [(astre, astre.position(t)) for astre in liste_astre]

        new_pos, new_vit = prochaine_pos_vit(pos, astres_positions, dt, vit)
        contacte, new_pos = est_en_contacte(pos, new_pos, projectile, astres_positions)
        
        if contacte is not None: # s'il y a un contacte
            projectile.ajouter_etat(new_pos, new_vit)  # ajoute la dernière position du projectile (sur l’astre)
            return contacte
        if new_pos[0] ** 2 + new_pos[1] ** 2 > distance_max**2:
            raise HorsSystemeError("Le projectile est hors du système.")
        if nombre_diteration_max < i: 
            raise TempsMaximumAtteinte("Le nombre de temps maximum a été atteint.")

        # Enregistre seulement les positions une fois par jour
        if i % Nb_iterations_enregistrement == 0 and i != 0:
            projectile.ajouter_etat(new_pos, new_vit)

        pos, vit = new_pos, new_vit
        t += dt
    return None
    


def acceleration( pos: list[float], astres_positions: list[tuple[AstreFixe, tuple[float, float]]]) -> np.ndarray:
    """_Calcule l'accélération (en UA/j²) que les astres actifs exercent sur le projectile a la position _pos__

    Args:
        pos (list[float]): _la position x et y du projectile (en UA)_
        astres_positions (list[tuple[AstreFixe, tuple[float,float]]]): _liste qui contient toutes les positions des astres et les astres fixes du système_

    Returns:
        np.ndarray: _accélération x et y du projectile (en UA/j²)_

    Levy : fonction écrite pour regrouper les deux boucles d'accélération de prochaine_pos_vit.
    """
    # Un astre inactif n'attire pas
    actifs = [(astre.masse, pos_astre) for astre, pos_astre in astres_positions if astre.actif]
    if not actifs:
        return np.zeros(2)
    # Tous les astres d'un coup avec numpy (une boucle Python par astre était le plus lent du programme)
    masses = np.array([masse for masse, _ in actifs])
    vecteurs_astre_projectile = np.array([pos_astre for _, pos_astre in actifs]) - pos
    distances2 = (vecteurs_astre_projectile ** 2).sum(axis=1)
    masses[distances2 < 1e-12] = 0  # Pour eviter les divisions par 0 : un astre au même point n'attire pas
    distances2 = np.maximum(distances2, 1e-12)

    # Fg = Gm1m2/d**2 et a = F/m1, donc a = G * m2/d**2 dans la direction vecteur / d, soit G * m2 * vecteur / d**3.
    # @ (produit matriciel) additionne les contributions de tous les astres : https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    return (G * masses / distances2 ** 1.5) @ vecteurs_astre_projectile


def prochaine_pos_vit( pos: list[float], astres_positions: list[tuple[AstreFixe, tuple[float, float]]], dt: float, v: list[float] ) -> tuple[tuple]:
    """_Calcule la prochaine vitesse et la prochaine position du projectile en utilisant
    l’algorithme de Verlet pour sa conservation d’énergie à long terme. (Pour plus d’information, consulte la vidéo suivante : https://www.youtube.com/watch?v=ii-N61PeeQU&t=747s)

    Args:
        pos (list[float]): _la position x et y du projectile à ce moment (en UA)_
        astres_positions (list[tuple[AstreFixe, tuple[float,float]]]): _liste qui contient toutes les positions des astres et les astres fixes du systeme_
        dt (float): _pas de temps (en j)_
        v (list[float]): _vitesse du projectile (en UA/j)_

    Returns:
        _tuple_ (new_pos, new_vit) :
            _new_pos_ -> _nouvelle position x et y du projectile (en UA)_

            _new_vit_ -> _nouvelle vitesse x et y du projectile (en UA/j)_
    """
    # 1- Trouvons a(t) -> a
    a = acceleration(pos, astres_positions)

    # 2- Trouvons la position au temps t + dt -> new pos
    new_pos = pos + v * dt + 0.5 * a * dt**2

    # 3- Trouvons a(t + dt) -> new_a
    new_a = acceleration(new_pos, astres_positions)

    # 4- Trouvons v(t+dt) -> new_v
    new_v = v + 1 / 2 * (new_a + a) * dt

    return new_pos, new_v


def est_en_contacte( pos: list[float], new_pos: list[float], p: Projectile, astres_positions: list[tuple[AstreFixe, tuple[float, float]]]) -> AstreFixe:
    """_Vérifie s'il est en contact ou non en calculant si la trajectoire des deux points a des points avec l’astre le plus proche_

    Args:
        pos (list[float]): _Position x et y du projectile au moment précédant (en UA)_
        new_pos (list[float]): _Nouvelle position x et y du projectile maintenant (en UA)_
        p (Projectile): _Projectile lancé_
        astres_positions (list[tuple[AstreFixe, tuple[float,float]]]): _Liste qui contient toutes les positions des astres et les astres fixes du système_

    Returns:
        AstreFixe: _l'astre touché et la position où elle a toucher OU None s'il ne touche rien_
    """

    # Trouver l'astre le plus proche du projectile
    astre_proche = None
    astre_proche_position = None
    distance_min = float("inf")
    for astre_position in astres_positions:
        pos_astre = astre_position[1]
        dx, dy = pos_astre - new_pos
        d2 = dx * dx + dy * dy

        if d2 < distance_min:
            distance_min = d2
            astre_proche = astre_position[0]
            astre_proche_position = pos_astre

    rayon_max = astre_proche.rayon + p.rayon

    if abs(new_pos[0] - pos[0]) < 1e-12:  # si division par 0
        return None, new_pos

    # 1- Trouvons la fonction f(x) = ax + b qui relit _pos_ et _new_pos_
    a = (new_pos[1] - pos[1]) / (new_pos[0] - pos[0])
    b = pos[1] - a * pos[0]

    # 2- Trouvons l'équation du cercle pour l’astre                      -> (x - astre_position[0])**2 + (y - astre_position[1] )**2 = r**2
    # 3- Ensuite, nous allons remplacer le y dans liquation
    # du cercle pour la fct f(x)                                        -> (x - astre_position[0])**2 + (ax + b - astre_position[1] )**2 = r**2
    # 4- avec des simplifications, nous avons                           -> (1+a**2)*x**2 + (2*a*b - 2*astre_position[0] - 2*a*astre_position[1])*x + (astre_position[0]**2 + astre_position[1]**2 + b**2 - 2*b*astre_position[1] - rayon_max**2) = 0
    A = 1 + a**2
    B = 2 * a * b - 2 * astre_proche_position[0] - 2 * a * astre_proche_position[1]
    C = (
        astre_proche_position[0] ** 2
        + astre_proche_position[1] ** 2
        + b**2
        - 2 * b * astre_proche_position[1]
        - rayon_max**2
    )

    # 5- Par la formule du discriminant                                 -> discriminant = B**2 - 4AC
    discriminant = (B) ** 2 - 4 * (A) * (C)

    # Les tangences produisent parfois de faux positifs
    # à cause des trajectoires quasi verticales.
    if discriminant <= 0:
        return None, new_pos

    # 6- par la formule quadratique                                     ->  (x = (-B +_ sqrt(discriminant)) / (2*A))
    x1, x2 = float((-B + np.sqrt(discriminant)) / (2 * A)), float((-B - np.sqrt(discriminant)) / (2 * A))

    # 7- par l'équation f(x)                                             -> (f(x) = ax + b)
    y1, y2 = float(a * x1 + b), float(a * x2 + b)

    # 8- Calcule le rectangle englobant du segment [pos, new_pos].
    # Une intersection avec le cercle n'est valide que si elle
    # se trouve à l'intérieur de ces bornes.
    xmax, xmin = max(pos[0], new_pos[0]), min(pos[0], new_pos[0])
    ymax, ymin = max(pos[1], new_pos[1]), min(pos[1], new_pos[1])

    # 9- regarde si le point passant a f(x) et liquation du cercle est entre les deux positions _pos_ et _new_pos_
    condition1 = (max(xmin, min(x1, xmax)) == x1 and max(ymin, min(y1, ymax)) == y1)  # -> si xmin <= x1 <= xmax et ymin <= y1 <=ymax
    condition2 = (max(xmin, min(x2, xmax)) == x2 and max(ymin, min(y2, ymax)) == y2)  # -> si xmin <= x2 <= xmax et ymin <= y2 <=ymax

    if condition1:
        return astre_proche, (x1, y1)

    elif condition2:
        return astre_proche, (x2, y2)
    return None, new_pos


""" ========================================================== DEBUG ==========================================================  """
# Test pour visualiser un projectile dans matplotlib : se lance seulement avec « python -m src.physique », pas a l'import
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pos_depart = None  # si None -> met la position du projectile a droite de la terre dune distance de 2* le rayon de la terre
    vit_depart = [
        0,
        0.01,
    ]  # [0.01, 0] == contact avec soleil, [0.01,0.01] == touche jamais rien, [0.00012,-0.017] == a peu pres le rond de la terre, [0,0] touche la terre

    # Creation des astres
    soleil = AstreFixe(
        nom="Soleil",
        masse=333000,
        couleur="jaune",
        taille=20,
        rayon=0.005,
        distance=0,
        periode=0,
    )

    terre = AstreFixe(
        nom="Terre",
        masse=1,
        couleur="bleu",
        taille=8,
        rayon=0.00005,
        distance=1.0,
        periode=365.25,
    )

    # Position de départ
    if pos_depart == None:

        pos_terre = terre.position(0)

        # Décale le projectile juste en dehors de la Terre
        direction = pos_terre / np.linalg.norm(pos_terre)

        pos_depart = pos_terre + direction * terre.rayon * 2

    # Projectil
    projectile = Projectile(
        nom="Meteor",
        masse=0.001,
        couleur="red",
        taille=3,
        rayon=0.00001,
        pos0=pos_depart,
        vit0=vit_depart,
        temps_apparition=0,
        duree_vie=1000,
    )

    impact = None
    try:
        impact = calculer_trajectoire(
            projectile,
            [soleil, terre],
            200,
        )
    except (TempsMaximumAtteinte, HorsSystemeError) as e:
        print("O NON :( : ", e, "\n")


    if impact is None:
        print("aucun impact")
    else:
        print(f"impact sur : {impact.nom}")
    print(f"nombre detats : {len(projectile.historique)}")

    print("\netat initial :")
    print(projectile.historique[0])

    print("\netat final :")
    print(projectile.historique[-1])

    # trasse la trajectoire du projectile
    x = [position[0] for position in projectile.historique]
    y = [position[1] for position in projectile.historique]

    # Trasse la terre pour 1 ans
    x_terre = [terre.position(i)[0] for i in range(int(366))]
    y_terre = [terre.position(i)[1] for i in range(int(366))]

    # extremome de la Terre selon la rayon (le bleu dans le graphique)
    x_rond = []
    for pos in x_terre:
        x_rond.extend([pos, pos + terre.rayon, pos, pos - terre.rayon])
    y_rond = []
    for pos in y_terre:
        y_rond.extend([pos + terre.rayon, pos, pos - terre.rayon, pos])

    x_soleil = [soleil.position(i)[0] for i in range(int(366))]
    y_soleil = [soleil.position(i)[1] for i in range(int(366))]

    plt.plot(x, y, "o--", color="m", label=projectile.nom)
    plt.plot(x_rond, y_rond, "o")

    plt.plot(x_terre, y_terre, "o-.", color="g", label=terre.nom)
    plt.plot(x_soleil, y_soleil, "o-.", color="y", label=soleil.nom)
    plt.legend()

    plt.show()