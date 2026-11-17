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
        prochain_t (int) : _Prochain temps a calculer (borne) ex: calculer jusqua temps 10_
        dt (float, optional): _pas de temps entre les positions (en jour)_. Initialiser à 0.1 (recommandé pour la rapidité) si plus précis et un peux moins rapide -> 0.01.
        temps_maximum (int, optional): _nombre de position calculée maximum_. Initialisé à 3653 jours (10 ans).

    Raises:
        HorsSystemeError: _si le projectile est hors du système._
        TempsMaximumAtteinte: _si le nombre de temps maximum a été atteint._

    Returns:
        AstreFixe: _retourne l'astre fixe seulement s'il a eu contacte avec un astre fixe_
    """
    
    if prochain_t - projectile.temps_apparition < 0: # verifi si le temps a calculer est un temps ou le projectile existe
        return
    
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

    
    for i in range(int(t/dt), int((prochain_t + 1)/dt)):#'''1 = buffer pour que ca marche''' 
        
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

    if(astre_proche is None):
        print(f"{distance_min = }")
        print(f"{astres_positions = }")

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

import pandas as pd
def trajectoire_optimal(projectile: Projectile, astre_debut: AstreFixe, astre_fin: AstreFixe, jour_lancement:int, liste_astre: list[AstreFixe], jour_max_lancement: int = 365, dt = 0.1):
	
    for vitesse in np.linspace(0.001, 0.05, num=10):
        angle_selon_astre_cibler_vitesse((0,2 * np.pi),projectile, 0.05, astre_debut, astre_fin, jour_lancement, liste_astre, jour_max_lancement, dt)
        return
        
    # trouve la vitesse minimal, avec le temps le plus vite, langle de projection avec l'energie la plsu bas avec un timer plsu bas que jour_max_lancement
     
    
    # pour l'energie -> 1/2 * vit**2 <=> 1/2 *vit**2 car m part
    # pour l'energie convertire les unitees en bonne unitee avec les unitee dans astre.py
    pos_debut = astre_debut.position(jour_lancement)
    
    # projectile serait la pour la masse a date:(
    angles = np.linspace(0, 2*np.pi, num=5)# 15 a 30
    vitesses = np.linspace(0.001, 0.05, num=10)# 20 # peux essayer de demander a lutilisateur un range 
    Nb_iterations_enregistrement = int(1 / dt) # calcule qui determine le nombre iteration pour une journee
    
    distances_min = np.zeros(5 * 10)
    
    # determine langles et la vitesse ou le projectile a ete le plus proche de lastre_fin par la distance entre eux a chaque iteration
    DEBUG= 0 
    it = 0
    
    
    angles_tableau = []
    vitesse_tableau = np.zeros(5*10)
    distance_min_tabelau = []
    temps_tableau = np.zeros(5*10)
    
    for angle in angles:
        vecteur_unitere = np.array([np.cos(angle), np.sin(angle)])
        for vitesse in vitesses:
            vit, pos = vitesse * vecteur_unitere, astre_debut.position(jour_lancement)
            dx, dy = pos_debut - astre_fin.position(jour_lancement)
            distance_min_base = np.sqrt(dx * dx + dy * dy)
            distance_min = distance_min_base
            jours = jour_lancement
            
            for i in range(int(jour_lancement/dt),int(jour_max_lancement/dt)):
                # Enregistre les coordonnées des astres ici, ce qui nous évite de les chercher constamment.

                astres_positions = [(astre, astre.position(jours)) for astre in liste_astre]
                
                pos, vit = prochaine_pos_vit(pos,astres_positions, dt, vit)
            
                distance = np.linalg.norm([pos, astre_fin.position(jours)])
                
                if distance < distance_min:
                    distance_min = distance
                    temps_tableau[it] = jours

                if i% Nb_iterations_enregistrement == 0:
                    jours += 1
                
            angles_tableau.append(angle)
            vitesse_tableau[it] = vitesse
            distance_min_tabelau.append(distance_min if distance_min < distance_min_base else np.inf)
            if distance_min >= distance_min_base:
                temps_tableau[it] = np.inf
            
            distances_min[it] = (distance_min if distance_min < distance_min_base else np.inf)
            it += 1
        print("angle numero: ", DEBUG + 1)
        DEBUG += 1
    print(np.sort(distances_min))
    
    print(len(distance_min_tabelau), len(vitesse_tableau), len(1/2 * projectile.masse * np.square(vitesse_tableau)), len(angles_tableau), len(temps_tableau))
    
    df = pd.DataFrame({
        "distance": distance_min_tabelau,
        "vitesse": vitesse_tableau,
        "Energie": 1/2 * projectile.masse * np.square(vitesse_tableau),
        "angle": angles_tableau,
        "temps": temps_tableau
    })
    
    print(df.sort_values("distance", ascending=True), '\n', df.sort_values("temps", ascending=True),'\n', df.sort_values("Energie", ascending=True))
    # 1- trouver le nombre dessaye avec un angle et une vitesse de debut (genre 100 fois experience) et trouver eu qui sont le plus proche 
	    
	# distance min 
    # distance_min = np.min(distances_parcourus_en_array - np.abs(astre_fin.distance() - astre_debut.distance()))
	    
	    
    # 2- trouver une zone qui serait plus petite exemple a la place des bon de 3 degrer, si on a les plus proche on va faire des bon de 0.5 degrer a la place


#def obserbe_trajectoire_astre(df:pd.DataFrame, distance_min: float, )


def angle_selon_astre_cibler_vitesse(range_angle: tuple[float,float], projectile: Projectile, vit_0: tuple, astre_debut: AstreFixe, astre_fin: AstreFixe, jour_lancement:int, liste_astre: list[AstreFixe], jour_max_lancement: int = 3652, dt = 0.1):
    pos_debut = astre_debut.position(jour_lancement)
    print('\n\n', np.sqrt(range_angle[0]**2 + range_angle[1]**2), '\n\n')
    if(np.sqrt(range_angle[0]**2 + range_angle[1]**2)) <= 0.1:
        
        contact = calculer_trajectoire(projectile, liste_astre, 3562)
        return 
        while contact == astre_fin:
            print("yesse sa marche.\n")
    
    pas = 20
    
    # projectile serait la pour la masse a date:(
    angles = np.linspace(range_angle[0] if range_angle[0] < range_angle[1] else range_angle[1], range_angle[1] if range_angle[1] > range_angle[0] else range_angle[0], num=pas)# 15 a 30
    Nb_iterations_enregistrement = int(1 / dt) # calcule qui determine le nombre iteration pour une journee
        
    distances_min = np.zeros(pas)
        
    # determine langles et la vitesse ou le projectile a ete le plus proche de lastre_fin par la distance entre eux a chaque iteration
    DEBUG= 0 
    it = 0
        
     
    angles_tableau = []
    distance_min_tabelau = []
    
    temps_tableau = np.zeros(pas)
    
    for angle in angles:
        vecteur_unitere = np.array([np.cos(angle), np.sin(angle)])
        
        vit, pos = vit_0 * vecteur_unitere, astre_debut.position(jour_lancement)
        dx, dy = pos_debut - astre_fin.position(jour_lancement)
        distance_min_base = np.sqrt(dx * dx + dy * dy)
        distance_min = distance_min_base
        jours = jour_lancement
                
        for i in range(int(jour_lancement/dt),int(jour_max_lancement/dt)):
                # Enregistre les coordonnées des astres ici, ce qui nous évite de les chercher constamment.
    
            astres_positions = [(astre, astre.position(jours)) for astre in liste_astre]
                    
            pos, vit = prochaine_pos_vit(pos,astres_positions, dt, vit)
                
            distance = np.linalg.norm([pos, astre_fin.position(jours)])
                    
            if distance < distance_min:
                distance_min = distance
                temps_tableau[it] = jours
    
            if i% Nb_iterations_enregistrement == 0:
                jours += 1
        
        angles_tableau.append(angle)
       
        distance_min_tabelau.append(distance_min if distance_min < distance_min_base else np.inf)
        if distance_min >= distance_min_base:
            temps_tableau[it] = np.inf
            
        #distances_min[it] = ((distance_min if distance_min < distance_min_base else np.inf), angle)
        it += 1
        #print("angle numero: ", DEBUG + 1)
        #DEBUG += 1
        
    #print(np.sort(distances_min))
    
        
    
    df = pd.DataFrame({
        "Angle": angles_tableau,
        "Distance": [distance_min_tabelau[i] * temps_tableau[i] for i in range(pas)]
    })
    df = df.sort_values("Distance", ignore_index=True)
    
    print('\n\n',df.head(), "\n", range_angle[0] if range_angle[0] < range_angle[1] else range_angle[1], " -> " ,range_angle[1] if range_angle[1] > range_angle[0] else range_angle[0])
    angle_selon_astre_cibler_vitesse((df.iloc[0]["Angle"], df.iloc[1]["Angle"]), projectile, vit_0, astre_debut, astre_fin, jour_lancement, liste_astre, jour_max_lancement, dt)
    return np.sort(distances_min) 
    
    






def depasser(pos, pos_astre_cible: AstreFixe, vers_interieurs: bool):
    if vers_interieurs:
        return np.linalg.norm(pos) <= np.linalg.norm(pos_astre_cible)
    elif not vers_interieurs:
        return np.linalg.norm(pos) >= np.linalg.norm(pos_astre_cible)
    
def dot(vecteur1, vecteur2):
    return np.dot(vecteur1, vecteur2).item()

def angle_optimal(astre_debut: AstreFixe, astre_cible: AstreFixe, vitesse_initiale: np.array[2], temps_aparition: int, liste_astre: list[AstreFixe], jour_max_lancement: int = 180, dt = 0.1):
    pos_debut, pos_cible = astre_debut.position(temps_aparition), astre_cible.position(temps_aparition)
    
    print(f"{pos_debut =}, {pos_cible = } ")
    print(np.arctan2(np.array([pos_cible[1] - pos_debut[1]]), np.array([pos_cible[0] - pos_debut[0]])))
    
    radian_min = np.arctan2(np.array([pos_cible[1] - pos_debut[1]]), np.array([pos_cible[0] - pos_debut[0]])).item()
    angle_min = np.degrees(radian_min)
    
    print(f"x -> {[pos_cible[1] - pos_debut[1]]}, y -> {[pos_cible[0] - pos_debut[0]]}")
    print(f"{angle_min = }")
    
    
    distance_nouvelle = float('nan')
    distance_avant = float('nan')
    
    distance_se_raproche = False
    
    temps_nouvelle = float('nan')
    temps_avant = float('nan')
    temps_diminue = False
    
    radian = radian_min
    
    distance_radian_temps_le_plus_petit = (float("inf"), float('nan'), float('nan'))
    
    
    # prendre en compte que plus quil va toucher, moins de temps sa va prendre, mem logique que distance_avant et diatance_nouvelle pour le temps
    # verifier sil prend plus de temps que les essaye avant\
    print(angle_min <= np.rad2deg(angle_min + 359),  ((distance_avant >= distance_nouvelle or not distance_se_raproche),  (temps_avant >= temps_nouvelle or not temps_diminue)))
    while radian <= np.deg2rad(angle_min + 90): #and ((distance_avant >= distance_nouvelle or not distance_se_raproche)): #and (temps_avant >= temps_nouvelle or not temps_diminue)): # modifier come cela ca marche selement avec les radians pas de rad2deg
        print(np.rad2deg(radian)%360)
    
        #print(np.degrees(radian).item(), f"{distance_avant = }", f"{distance_nouvelle = }", f"{temps_avant = }", f"{temps_nouvelle = }")
        if distance_avant >= distance_nouvelle and not distance_se_raproche:
            distance_se_raproche = True
        if temps_avant >= temps_nouvelle and not temps_diminue:
            temps_diminue = True
            
        distance_avant = distance_nouvelle
        temps_avant = temps_nouvelle
        
        vecteur_unitere = np.append(np.cos(radian), np.sin(radian)) # TODO: Changer cela c surS
        
        # pour le dot product je l'ai apris par: https://www.youtube.com/watch?v=nXrEX6j-Mws&t=513s
        pos, vit = astre_debut.position(temps_aparition), vitesse_initiale * vecteur_unitere
        jours = temps_aparition
        
        astres_positions = [(astre, astre.position(jours)) for astre in liste_astre]
        #print(f"\n { radian_min = }")
        #print(f" {np.append(np.cos(radian_min), np.sin(radian_min)) = }")
        #print(f" {dot(np.append(np.cos(radian_min), np.sin(radian_min)), pos)}")
        #print(f" { radian_min = } \n")
        
        #if astre_debut.distance() > astre_cible.distance():
            
        
        
        
        vers_interieur = np.linalg.norm(pos_debut) < np.linalg.norm(pos_cible)
        
        
        #print(np.append(np.cos(radian_min), np.sin(radian_min)), f"     {pos = }    ", f" {dot(np.append(np.cos(radian), np.sin(radian)), pos) = }")
        vecteur_cible_position = pos - astre_cible.position(jours) 
        
        # dot(np.append(np.cos(radian_min), np.sin(radian_min)), vecteur_cible_position) >= 0
        
        distance = np.linalg.norm(pos - astre_cible.position(jours))
        new_distance = 0
        while jours <= jour_max_lancement: #depasser(pos, astre_cible.position(jours), vers_interieur) and jours <= jour_max_lancement:# TODO: Changer cela c surS
            distance = np.linalg.norm(pos - astre_cible.position(jours))
            astres_positions = [(astre, astre.position(jours)) for astre in liste_astre]
            pos, vit = prochaine_pos_vit(pos, astres_positions, dt, vit)
            vecteur_cible_position = pos - astre_cible.position(jours) 
            jours += dt
            new_distance = np.linalg.norm(pos - astre_cible.position(jours))
            if distance_radian_temps_le_plus_petit[0] >= new_distance.item():
                print(distance_radian_temps_le_plus_petit[0], new_distance, distance_radian_temps_le_plus_petit[0] >= new_distance)
                distance_radian_temps_le_plus_petit = (new_distance, radian, jours)
                print(distance_radian_temps_le_plus_petit)
            
            # ajout de la colision
        
        radian += np.deg2rad(0.01)
        #print("---------------------------------------------------------\n")
        #print(f" {dot(np.append(np.cos(radian_min), np.sin(radian_min)), vecteur_cible_position) = }")
        #print(f" {np.append(np.cos(radian_min), np.sin(radian_min)) = }")
        #print(f" {pos = }")
        #print(jours, jour_max_lancement)
        #print("---------------------------------------------------------\n")
        
        #if jours >= jour_max_lancement:
        #    print("hi:)")
        #    continue
        
        
        distance_entre_astre = np.linalg.norm([pos - astre_cible.position(jours)])
        distance_nouvelle = distance_entre_astre

        temps_nouvelle = jours
        
        
    #print(np.degrees(radian).item(), f"{distance_avant = }", f"{distance_nouvelle = }", f"{temps_avant = }", f"{temps_nouvelle = }")
            
    #print(angle_min <= np.rad2deg(angle_min + 359),  ((distance_avant >= distance_nouvelle or not distance_se_raproche),  (temps_avant >= temps_nouvelle or not temps_diminue)))
    
    if(radian == radian_min):
        while True:
            pass
    
    print(f"{distance_radian_temps_le_plus_petit = }")
    
    return distance_radian_temps_le_plus_petit[1], distance_radian_temps_le_plus_petit[2]


# Le problème de Lambert

# 3.1.1. Computational Algorithm 1 de la page 92
def C(psi, H_0 = 1/2, eta = 3, tolerance = 1e-12, M = 100):
    """_Calcule la valeur psi selon la serie de Stumpff
        Detail -> https://www.academia.edu/103281934/Lambert_Universal_Variable_Algorithm pdf_

    Args:
        psi (_float_): _variable que l'algorithme cherche_
        H_0 (_float_, optional): _constante pour le Cn(ψ)_. Defaults to 1/2 pour C2.
        eta (int, optional): _constante pour le Cn(ψ)_. Defaults to 3 pour C2.
        tolerance (_float_, optional): _tolerance derreur_. Defaults to 1e-12.
        M (int, optional): _nombre pour limiter les iterrations dans la serie de Stumpff_. Defaults to 100.

    Returns:
       C _float_: _description_
    """    
    H = C = H_0
    K = 2
    L = eta
    N = None
    
    for i in range(1, M + 1):
        H = - psi * H / (2 * K * L)
        if np.abs(H) <= tolerance:
            N = i
            break
        
        C += H 
        K += 1
        L += 2
    if N is None:
        raise ValueError("Physique - C() : N is None. ")
    return C


def C2_and_C3_serie(psi):
    C2 = C(psi, H_0=1/2, eta=3)
    C3 = C(psi, H_0=1/6, eta=5)
    return C2, C3 

# 3.1.2. Computational Algorithm 2 
def C2_and_C3(psi, psi_max):
    """_summary_

    Args:
        psi (_type_): _description_
        psi_max (_type_): _valeur maximale de ψ pour laquelle on accepte d'utiliser directement la série de Stumpff_

    Raises:
        ValueError: _description_
    """
    
    K = 0 
    
    # etape 2-5
    while np.abs(psi) > psi_max:
        # print(f"{K = }, {psi = }, {psi_max = }")
        K += 1 
        psi /= 4
    C2, C3 = C2_and_C3_serie(psi)
    
    # etape 6-11
    while not K == 0:
        K -= 1
        new_C2 = 0.5 * (1 - psi * C3) ** 2 # (2.18)
        new_C3 = (C2 + C3 - psi * C3 * C2)/ 4 # (2.17)
        
        C2 = new_C2
        C3 = new_C3
        psi *= 4
    return C2, C3


#  3.2.1. Computational Algorithm 3
#  solution of a Lambert problem using the bisection technique
def problematique_de_Lambert(vec_r_0, vec_r, temps_voulu, masse_astre_central, psi_0, psi_max = 4 * np.pi**2, psi_min = -4 * np.pi**2, tm = 1, M = 100,  tolerance = 1e-12):
    """_summary_

    Args:
        vec_r_0 (_type_): _le vecteur de la positionn inicial_
        vec_r (_type_): _le vecteur de la position final_
        temps_voulu (_type_): _le temps entre le l'ancement et l'impact_
        masse_astre_central (_type_): _la masse de lastre central (pour calculer mu)_
        psi_0 (_type_): _la valeur de psi inicial_
        psi_max (_type_): _la valeur max de psi_
        psi_min (_type_): _la valeur min de psi_
        tm (float): _deside si choisir le chemain le plus long(tm = -1) ou plus cour(tm = +1)_
        M (_type_): _nombre pour limiter les iterrations dans la serie de Stumpff. recomander 100_ 
        tolerance (_type_): _tolerance des insertitudes_
        mu (_type_): _gravitational parameter (universal gravitational constant times the sum of the two masses)_
    """
    
    
    # etape 1-5
    r_0 = np.linalg.norm(vec_r_0)
    r = np.linalg.norm(vec_r)
    
    if any([r_0 == 0, r == 0]):
        raise ValueError("Physique - problematique_de_Lamber(): il y a une donnee a la position (0,0)")
    
    gamma = np.dot(vec_r_0, vec_r)/(r * r_0)
    beta = tm*(1- gamma**2)**0.5
    A = tm * (r*r_0*(1 + gamma))**0.5
    mu = G * masse_astre_central
    psi_m = 0.1
    
    print(f"{vec_r_0 = }, {vec_r = }")
    print(f"{gamma = }")
    
    # etape 6
    if A == 0:
        raise ValueError("Physique - problematique_de_Lambert(): on ne peux pas calculer l'orbitre. ") # TODO essayer de faire une chose comme dans calculer trajectoire
    
    # etape 7
    for i in range(1, M + 1):
        # print(f"{psi_0 = }, {psi_min = }, {psi_max = }")
        psi = psi_0
        C2, C3 = C2_and_C3(psi, psi_m)
        B = r_0 + r + (1/(C2)*0.5) * (A*(psi*C3 - 1))
        
        if A > 0 and B < 0:
            while B <= 0 :
                
                psi_min = psi
                psi = psi_0 = (psi_max + psi)/2
                C2, C3 = C2_and_C3(psi, psi_m)
                B = r_0 + r + (1/(C2)*0.5) * (A*(psi*C3 - 1))
        
        chi = (B/C2)**0.5
        temps_calculer = (1/(mu)**0.5)*(chi**3 * C3 + A * B**0.5) 
        
        print(
            f"it={i}, psi={psi:.6f}, "
            f"temps={temps_calculer:.6f}, "
            f"voulu={temps_voulu:.6f}"
        )
        
        if np.abs(temps_voulu - temps_calculer) < tolerance: 
            break
        if temps_calculer <= temps_voulu :
            psi_min = psi
        else:
            psi_max = psi
        psi_0 = psi = (psi_max + psi_min)/2
    
    # etape 8 
    # Multiplicateurs de Lagrange -> https://gereshes.com/2018/06/25/lagrange-fg-solution-the-2-body-problem/
    F_fct = 1 - B/r_0
    G_fct = A * (B/mu) **0.5
    
    vitesse_depart = (1/G_fct) * (vec_r - (vec_r_0*F_fct))
    return vitesse_depart

    
    
    
    
    
    
    
    # pour la definition de e et des relations entre les grandeur caracteristique d'une hyperbole -> https://fr.wikipedia.org/wiki/Hyperbole_(math%C3%A9matiques)


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
        distance=1e-32,
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
    
    mars = AstreFixe(
            nom="Mars",
            masse=0.107,
            couleur="darkorange",
            taille=0.533,
            rayon=0.00002270219,
            distance=1.52,
            periode=687.0,
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
        
        temps_apparition = 40
        astre_debut = mars
        astre_cible = terre
        
        
        #terre.position(temps_apparition)
        vitesse_pour_hit = problematique_de_Lambert(astre_debut.position(0), astre_cible.position(temps_apparition), temps_apparition, soleil.masse, 0)
        print("-------------------------Resultat----------------------------\n")
        print(f"{vitesse_pour_hit = }")
        print(f"{pos_depart = }")
        
        direction  = (vitesse_pour_hit)/np.linalg.norm(vitesse_pour_hit)
        rayon_projectile = 0.00001
        
        pos_depart_t = astre_debut.position(0)
        pos_depart = pos_depart_t + direction * astre_debut.rayon + direction * rayon_projectile
        
        vitesse_pour_hit = problematique_de_Lambert(pos_depart, astre_cible.position(temps_apparition), temps_apparition, soleil.masse, 0)
        print("-------------------------Resultat----------------------------\n")
        print(f"{vitesse_pour_hit = }")
        print(f"{pos_depart = }")
        
        projectile = Projectile(
                nom="Meteor",
                masse=0.001,
                couleur="red",
                taille=3,
                rayon=rayon_projectile,
                pos0=pos_depart,
                vit0=vitesse_pour_hit,
                temps_apparition=0,
                duree_vie=1000,
            )
        
        #print(f"\n\n\n {terre.position(temps_apparition) + 0.0001* vecteur_unitere + 0.005* vecteur_unitere}")
        #print(np.rad2deg(radian))
        #print(f"{temps_apparition = }")
        #print(f"{temps_nouvelle = }")
        
        #\impact = trajectoire_optimal(
        #    projectile, terre, soleil, 0,
        #    [soleil, terre],
        #)
        #temps_pour_terre_DEBUG = temps_nouvelle
        impact = calculer_trajectoire(projectile, [soleil, terre], int(temps_apparition * 2))
        
    except (TempsMaximumAtteinte, HorsSystemeError) as e:
        print("O NON :( : ", e, "\n")


    if impact is None:
        print("aucun impact")
    else:
        print(f"impact sur : {impact.nom}")
    print(f"nombre detats : {len(projectile.historique) - 1}")

    print("\netat initial :")
    print(projectile.historique[0])

    print("\netat final :")
    print(projectile.historique[-1])

    # trasse la trajectoire du projectile
    x = [position[0] for position in projectile.historique]
    y = [position[1] for position in projectile.historique]

    # Trasse la terre pour 1 ans
    x_terre = [terre.position(i)[0] for i in range(int(temps_apparition + 1))]
    y_terre = [terre.position(i)[1] for i in range(int(temps_apparition + 1))]

    # extremome de la Terre selon la rayon (le bleu dans le graphique)
    x_rond = []
    for pos in x_terre:
        x_rond.extend([pos, pos + terre.rayon, pos, pos - terre.rayon])
    y_rond = []
    for pos in y_terre:
        y_rond.extend([pos + terre.rayon, pos, pos - terre.rayon, pos])
        
    # Trasse la terre pour 1 ans
    x_terre = [terre.position(i)[0] for i in range(int(temps_apparition + 1))]
    y_terre = [terre.position(i)[1] for i in range(int(temps_apparition + 1))]
    
    x_mars = [mars.position(i)[0] for i in range(int(temps_apparition + 1))]
    y_mars = [mars.position(i)[1] for i in range(int(temps_apparition + 1))]


    x_soleil = [soleil.position(i)[0] for i in range(int(366))]
    y_soleil = [soleil.position(i)[1] for i in range(int(366))]

    plt.plot(x, y, "o--", color="r", label=projectile.nom)
    plt.plot(x_rond, y_rond, "o")
    
    plt.plot(x, y, "o--", color="m", label=projectile.nom)
    plt.plot(x_mars, y_mars, "o")

    plt.plot(x_terre, y_terre, "o-.", color="g", label=terre.nom)
    plt.plot(x_soleil, y_soleil, "o-.", color="y", label=soleil.nom)
    plt.legend()

    plt.show()