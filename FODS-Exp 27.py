import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# Read CSV file
# ---------------------------------------

data = pd.read_csv("soccer_players.csv")

print("Complete Soccer Players Dataset")
print("--------------------------------")
print(data)

# ---------------------------------------
# Top 5 players by number of goals
# ---------------------------------------

top_goals = data.sort_values(
    by="Goals",
    ascending=False
).head(5)

print("\nTop 5 Players with Highest Goals")
print("---------------------------------")
print(top_goals[["Name", "Position", "Goals"]])

# ---------------------------------------
# Top 5 players by weekly salary
# ---------------------------------------

top_salary = data.sort_values(
    by="Weekly_Salary",
    ascending=False
).head(5)

print("\nTop 5 Players with Highest Weekly Salary")
print("-----------------------------------------")
print(top_salary[["Name", "Position", "Weekly_Salary"]])

# ---------------------------------------
# Calculate average age
# ---------------------------------------

average_age = data["Age"].mean()

print("\nAverage Age of Players")
print("---------------------")
print("Average Age:", round(average_age, 2))

# ---------------------------------------
# Players above average age
# ---------------------------------------

above_average_age = data[data["Age"] > average_age]

print("\nPlayers Above Average Age")
print("-------------------------")
print(above_average_age[["Name", "Age", "Position"]])

# ---------------------------------------
# Distribution of players by position
# ---------------------------------------

position_count = data["Position"].value_counts()

print("\nNumber of Players by Position")
print("-----------------------------")
print(position_count)

# ---------------------------------------
# Bar Chart
# ---------------------------------------

plt.figure(figsize=(8, 5))

position_count.plot(
    kind="bar"
)

plt.title("Distribution of Soccer Players by Position")
plt.xlabel("Position")
plt.ylabel("Number of Players")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()