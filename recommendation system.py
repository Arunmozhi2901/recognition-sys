import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

class RecommendationSystem:
    def __init__(self, data):
        self.data = data
        self.user_item_matrix = self.create_user_item_matrix()
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)

    def create_user_item_matrix(self):
        return self.data.pivot(index='user_id', columns='book_title', values='rating').fillna(0)

    def get_user_recommendations(self, user_id, num_recommendations=3):
        user_index = self.data['user_id'].unique().tolist().index(user_id)
        similar_users = list(enumerate(self.similarity_matrix[user_index]))
        similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:]  # Exclude self

        recommendations = []
        for similar_user in similar_users:
            user_index = similar_user[0]
            user_ratings = self.user_item_matrix.iloc[user_index]
            recommendations.append(user_ratings[user_ratings > 0])

        recommendations = pd.concat(recommendations)
        recommendations = recommendations.groupby(recommendations.index).mean().sort_values(ascending=False)
        return recommendations.head(num_recommendations)

class RecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Book Recommendation System")
        self.root.config(bg="#F7F7F7")
        self.root.geometry("450x400")

        # Sample Data
        data = {
            'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            'book_title': [
                'The Silent Patient',
                'Where the Crawdads Sing',
                'The Midnight Library',
                'The Vanishing Half',
                'Daisy Jones & The Six',
                'Circe',
                'Anxious People',
                'The Invisible Life of Addie LaRue',
                'The Song of Achilles',
                'Project Hail Mary'
            ],
            'rating': [5, 4, 3, 5, 3, 4, 5, 2, 3, 4]
        }
        self.df = pd.DataFrame(data)
        self.recommender = RecommendationSystem(self.df)

        self.setup_ui()

    def setup_ui(self):
        title_label = tk.Label(self.root, text="📚 Book Recommendation", 
                                font=("Helvetica Neue", 20, 'bold'), bg="#F7F7F7", fg="#333")
        title_label.pack(pady=20)

        instruction_label = tk.Label(self.root, text="Enter User ID (1-5):", 
                                      font=("Helvetica Neue", 12), bg="#F7F7F7", fg="#555")
        instruction_label.pack(pady=10)

        self.user_input = tk.Entry(self.root, font=("Helvetica Neue", 16), bd=2, relief="solid")
        self.user_input.pack(pady=10)
        self.user_input.insert(0, "1")
        self.user_input.bind('<Return>', self.get_recommendations)

        self.submit_button = tk.Button(self.root, text="Get Recommendations", 
                                        command=self.get_recommendations, 
                                        font=("Helvetica Neue", 14), bg="#007BFF", fg="white", relief="raised")
        self.submit_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica Neue", 12), bg="#F7F7F7", fg="#333")
        self.result_label.pack(pady=20)

    def get_recommendations(self, event=None):
        try:
            user_id = int(self.user_input.get())
            if user_id not in self.df['user_id'].unique():
                raise ValueError("User ID not found.")

            recommended_items = self.recommender.get_user_recommendations(user_id)
            recommendations_str = "\n".join([f"{item} (Rating: {rating:.2f})" for item, rating in recommended_items.items()])
            messagebox.showinfo("Recommendations", f"Recommended books for User {user_id}:\n{recommendations_str}")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = RecommendationApp(root)
    root.mainloop()
