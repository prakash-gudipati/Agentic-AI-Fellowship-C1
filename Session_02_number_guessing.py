import random

print("=" * 50)
print("Welcome to the number guessing game")
print("=" * 50)

print("\n Choose the difficulty level")
print("Easy    Numbers(1-10)  5 Guesses")
print("Medium  Numbers(1-50)  7 Guesses")
print("Hard    Numbers(1-100) 10 Guesses")

difficulty = input("\n Enter diffculty (easy/medium/hard): ").lower()

if difficulty == "easy":
    min_number = 1
    max_number = 10
    max_guesses = 5
elif difficulty == "medium":
    min_number = 1
    max_number = 50
    max_guesses = 7
elif difficulty == "hard":
    min_number = 1
    max_number = 100
    max_guesses = 10
else:
    print("\n Invalid choice. Defaulting to easy mode")
    min_number = 1
    max_number = 10
    max_guesses = 5

secret_number = random.randint(min_number, max_number)

attempt_count = 0
player_won = False

all_guesses = []

while attempt_count < max_guesses:
    attempt_count = attempt_count + 1

    player_guess = int(input(f"\n Attempt {attempt_count}/{max_guesses} - Enter your guess:"))
    all_guesses.append(player_guess)

    if player_guess > secret_number:
        print("Too high!")
    elif player_guess < secret_number:
        print("Too Low!")
    else:
        print(f"Correct! You guess {secret_number} in {attempt_count} attemps. You Won!")
        player_won = True
        break

if not player_won:
    print(f"\nOut of guesses! The secret number was {secret_number}")

high_guesses = [guess for guess in all_guesses if guess > secret_number]

print(f"\n Your high guesses : {high_guesses}")
print("Game Over!! Thank you for playing")