#Zadanie 4
#Implementacja kółko krzyżyk
import math

class Plansza:
    def __init__(self):
        #pusta plansza to lista 9 spacji
        self.pola = [" " for _ in range(9)]

    #wyswitlamy tablice do gry
    def wyswietl(self):
        print(f"\n {self.pola[0]} | {self.pola[1]} | {self.pola[2]} ")
        print("---+---+---")
        print(f" {self.pola[3]} | {self.pola[4]} | {self.pola[5]} ")
        print("---+---+---")
        print(f" {self.pola[6]} | {self.pola[7]} | {self.pola[8]} \n")

    def wolne_ruchy(self):
        return [i for i, x in enumerate(self.pola) if x == " "]

    def czy_wygral(self, gracz):
        #lista wygrywających trójek (indeksy)
        win_kombinacje = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8), #poziomo
            (0, 3, 6), (1, 4, 7), (2, 5, 8), #pionowo
            (0, 4, 8), (2, 4, 6)             #ukośnie
        ]

        for a, b, c in win_kombinacje:
            if self.pola[a] == self.pola[b] == self.pola[c] == gracz:
                return True
        return False

    def czy_pelna(self):
        return " " not in self.pola


class GraAI:
    def __init__(self):
        self.plansza = Plansza()
        self.licznik_wywolan = 0
        
    def start(self):
        print("Kółko i Krzyżyk")
        print("Ty jesteś 'X', Komputer to 'O'.")
        self.plansza.wyswietl()

        while True:
            #ruch gracza
            while True:
                try:
                    wybor = int(input("Twój ruch (0-8): "))
                    if self.plansza.pola[wybor] == " ":
                        self.plansza.pola[wybor] = "X"
                        break
                    else:
                        print("Pole zajęte!")
                except (ValueError, IndexError):
                    print("Błędny numer!")

            self.plansza.wyswietl()
            if self.sprawdz_koniec(): break

            #ruch ai
            print("Komputer myśli...")
            najlepszy_ruch = self.znajdz_najlepszy_ruch()
            self.plansza.pola[najlepszy_ruch] = "O"
            
            self.plansza.wyswietl()
            if self.sprawdz_koniec(): break

    def sprawdz_koniec(self):
        if self.plansza.czy_wygral("X"):
            print("Wygrałeś!")
            return True
        if self.plansza.czy_wygral("O"):
            print("Komputer wygrał!")
            return True
        if self.plansza.czy_pelna():
            print("Remis!")
            return True
        return False

    #algorytm gry 
    def znajdz_najlepszy_ruch(self):
        self.licznik_wywolan = 0
        #startowo na nieskonczonosc 
        najlepszy_wynik = -math.inf
        #indeks pola narazie na -1, takiego ktorego nie ma
        najlepszy_ruch = -1
        
        #sprawdzamy wolne pola
        for ruch in self.plansza.wolne_ruchy():
            #robimy ruch (udawany, przymiarkowy)
            self.plansza.pola[ruch] = "O"
            
            #obliczamy jego wartość algorytmem alfa-beta, ocena sytuacji na planszy
            wynik = self.alfa_beta(0, -math.inf, math.inf, False)
            
            #cofamy ruch przymiarkowy (Backtracking) - 
            self.plansza.pola[ruch] = " "
            
            #porownanie ruchow
            if wynik > najlepszy_wynik:
                najlepszy_wynik = wynik
                najlepszy_ruch = ruch
        print(f"AI przeanalizowało: {self.licznik_wywolan} stanów gry.")
        return najlepszy_ruch

    #algorytm alfa beta
    def alfa_beta(self, glebokosc, alfa, beta, czy_maksymalizuje):
        self.licznik_wywolan += 1
        # Sprawdzenie stanów końcowych
        #czy w obecnym stanie wygralo AI czy ja?
        if self.plansza.czy_wygral("O"): return 10  #ai
        if self.plansza.czy_wygral("X"): return -10 #ja
        if self.plansza.czy_pelna(): return 0       #remis

        if czy_maksymalizuje: #tura ai (chce max)
            #szukamy max
            max_eval = -math.inf
            #spawdzamy ruchy
            for ruch in self.plansza.wolne_ruchy():
                #robi ruch
                self.plansza.pola[ruch] = "O"       
                #rekurencja
                eval = self.alfa_beta(glebokosc + 1, alfa, beta, False)
                #cofam ruch
                self.plansza.pola[ruch] = " "         
                #czy wynik lepszt od dotychczasowego
                max_eval = max(max_eval, eval)
                #aktualizacja alfa
                alfa = max(alfa, eval)
                if beta <= alfa: break  #cięcie
            return max_eval
        
        else: #moja tura (chce min dla komputera)
            #szukamy min
            min_eval = math.inf
            #spawrdzamy wolne pola
            for ruch in self.plansza.wolne_ruchy():
                #robimy ruch        
                self.plansza.pola[ruch] = "X"         
                eval = self.alfa_beta(glebokosc + 1, alfa, beta, True)
                #cofamy ruch
                self.plansza.pola[ruch] = " "
                #rekurencja
                min_eval = min(min_eval, eval)
                #beta - najgorszy wynik dla ai
                beta = min(beta, eval)
                if beta <= alfa: break #cięcie
            return min_eval
        
if __name__ == "__main__":
    gra = GraAI()
    gra.start()