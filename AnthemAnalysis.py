from tkinter import *
import tkinter as tk
import tkinter as ttk
import csv
from tkinter import messagebox

import customtkinter as ctk
import openai

# Data Structures
import numpy as np
import pandas as pd
import json
# Corpus Processing
import re
import nltk.corpus
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# K-Means
from sklearn import cluster
# Visualization and Analysis
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud
import geopandas as gpd

TotalWidth = 1550
TotalHeight = 900
global anthem_entry_var
global dfs


def CompareAnthems():
    global dfs
    ctk.CTkCanvas(win)
    screen = Canvas(master=win, bg='white', height=TotalHeight, width=TotalWidth)
    screen.place(x=0, y=0)
    compare_anthems = ctk.CTkFrame(master=screen, height=TotalHeight, width=TotalWidth, fg_color='white',
                                   corner_radius=0)
    compare_anthems.pack()

    info_listbox = None
    select_all_var = None
    all_countries = None
    search_entry = None

    def on_select(event):
        global dfs
        selected_indices = info_listbox.curselection()
        selected_countries = [info_listbox.get(idx) for idx in selected_indices]
        data = pd.read_csv('anthems.csv', encoding='utf-8')
        data.columns = map(str.lower, data.columns)
        continents = ['Europe', 'South_America', 'North_America', 'Asia', 'Africa', 'Oceania']
        data = data.loc[data['continent'].isin(continents)]
        data = data.loc[data['country'].isin(selected_countries)]
        num_rows = int(len(data))
        if num_rows < 8:
            messagebox.showwarning("Warning", 'Minimum selected countries should be 8')
            CompareAnthems()
            return
        print("Selected Countries:", selected_countries)
        print(data.head(6))
        corpus = data['anthem'].tolist()
        corpus[7][0:447]

        def removeWords(listOfTokens, listOfWords):
            return [token for token in listOfTokens if token not in listOfWords]

        def applyStemming(listOfTokens, stemmer):
            return [stemmer.stem(token) for token in listOfTokens]

        def twoLetters(listOfTokens):
            twoLetterWord = []
            for token in listOfTokens:
                if len(token) <= 2 or len(token) >= 21:
                    twoLetterWord.append(token)
            return twoLetterWord

        def processCorpus(corpus, language):
            stopwords = nltk.corpus.stopwords.words(language)
            param_stemmer = SnowballStemmer(language)
            countries_list = [line.rstrip('\n') for line in open('lists/countries.txt')]
            nationalities_list = [line.rstrip('\n') for line in
                                  open('lists/nationalities.txt')]
            other_words = [line.rstrip('\n') for line in
                           open('lists/stopwords_scrapmaker.txt')]

            for document in corpus:
                index = corpus.index(document)
                corpus[index] = corpus[index].replace(u'\ufffd', '8')
                corpus[index] = corpus[index].replace(',', '')
                corpus[index] = corpus[index].rstrip('\n')
                corpus[index] = corpus[index].casefold()

                corpus[index] = re.sub('\W_', ' ', corpus[index])
                corpus[index] = re.sub("\S*\d\S*", " ", corpus[
                    index])
                corpus[index] = re.sub("\S*@\S*\s?", " ", corpus[index])
                corpus[index] = re.sub(r'http\S+', '', corpus[index])
                corpus[index] = re.sub(r'www\S+', '', corpus[index])

                listOfTokens = word_tokenize(corpus[index])
                twoLetterWord = twoLetters(listOfTokens)

                listOfTokens = removeWords(listOfTokens, stopwords)
                listOfTokens = removeWords(listOfTokens, twoLetterWord)
                listOfTokens = removeWords(listOfTokens, countries_list)
                listOfTokens = removeWords(listOfTokens, nationalities_list)
                listOfTokens = removeWords(listOfTokens, other_words)

                listOfTokens = applyStemming(listOfTokens, param_stemmer)
                listOfTokens = removeWords(listOfTokens, other_words)

                corpus[index] = " ".join(listOfTokens)
                corpus[index] = unidecode(corpus[index])

            return corpus

        language = 'english'
        corpus = processCorpus(corpus, language)
        print(corpus[0][0:460])

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        tf_idf = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names_out())
        final_df = tf_idf
        row_for_k = ("{} rows".format(final_df.shape[0]))
        print(row_for_k)
        print(final_df.T.nlargest(5, 0))

        def run_KMeans(max_k, data):
            max_k += 1
            kmeans_results = dict()
            for k in range(2, max_k):
                kmeans = cluster.KMeans(n_clusters=k, init='k-means++', n_init=10, tol=0.0001, random_state=1,
                                        algorithm='lloyd')
                kmeans_results.update({k: kmeans.fit(data)})
            return kmeans_results

        def printAvg(avg_dict):
            for avg in sorted(avg_dict.keys(), reverse=True):
                print("Avg: {}\tK:{}".format(avg.round(4), avg_dict[avg]))

        def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
            fig, ax1 = plt.subplots(1)
            fig.set_size_inches(8, 6)
            ax1.set_xlim([-0.2, 1])
            ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
            ax1.axvline(x=silhouette_avg, color="red",
                        linestyle="--")
            ax1.set_yticks([])
            ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.title(("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight='bold')
            y_lower = 10
            sample_silhouette_values = silhouette_samples(df, kmeans_labels)
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                                  edgecolor=color, alpha=0.7)
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10
            plt.show()

        def silhouette(kmeans_dict, df, plot=False):
            df = df.to_numpy()
            avg_dict = dict()
            for n_clusters, kmeans in kmeans_dict.items():
                kmeans_labels = kmeans.predict(df)
                silhouette_avg = silhouette_score(df, kmeans_labels)  # Average Score for all Samples
                avg_dict.update({silhouette_avg: n_clusters})
                if (plot): plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg)

        # if(row_for_k<'8'):
        #     k=2
        # else:
        k = 8
        kmeans_results = run_KMeans(k, final_df)

        def get_top_features_cluster(tf_idf_array, prediction, n_feats):
            labels = np.unique(prediction)
            global dfs
            dfs = []
            for label in labels:
                id_temp = np.where(prediction == label)  # indices for each cluster
                x_means = np.mean(tf_idf_array[id_temp], axis=0)  # returns average score across cluster
                sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top 20 scores
                features = vectorizer.get_feature_names_out()
                best_features = [(features[i], x_means[i]) for i in sorted_means]
                df = pd.DataFrame(best_features, columns=['features', 'score'])
                dfs.append(df)
            return dfs

        def plotWords(dfs, n_feats):

            plt.figure(figsize=(8, 4))
            for i in range(0, len(dfs)):
                plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
                sns.barplot(x='score', y='features', orient='h', data=dfs[i][:n_feats])
                plt.tight_layout()
                plt.show()

        # best_result = max(kmeans_results.keys())

        # if(best_result >=5 ):
        best_result = 5

        kmeans = kmeans_results.get(best_result)

        final_df_array = final_df.to_numpy()
        prediction = kmeans.predict(final_df)
        n_feats = 20
        dfs = get_top_features_cluster(final_df_array, prediction, n_feats)
        print(dfs)
        plotWords(dfs, 13)

        def centroidsDict(centroids, index):
            a = centroids.T[index].sort_values(ascending=False).reset_index().values
            centroid_dict = dict()

            for i in range(0, len(a)):
                centroid_dict.update({a[i, 0]: a[i, 1]})

            return centroid_dict

        def generateWordClouds(centroids):
            wordcloud = WordCloud(max_font_size=100, background_color='white')
            for i in range(0, len(centroids)):
                centroid_dict = centroidsDict(centroids, i)
                wordcloud.generate_from_frequencies(centroid_dict)

                plt.figure()
                plt.title('Cluster {}'.format(i))
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.show(block=False)
                plt.get_current_fig_manager().window.state('zoomed')
            plt.show()

        centroids = pd.DataFrame(kmeans.cluster_centers_)
        centroids.columns = final_df.columns
        generateWordClouds(centroids)
        labels = kmeans.labels_
        data['label'] = labels
        data.head()
        geo_path = 'datasets/world-countries.json'
        country_geo = json.load(open(geo_path))
        gpf = gpd.read_file(geo_path)

        # Merging on the alpha-3 country codes
        merge = pd.merge(gpf, data, left_on='id', right_on='alpha-3')
        data_to_plot = merge[["id", "name", "label", "geometry"]]

        data_to_plot.head(3)

    def toggle_select_all():
        select_all_state = select_all_var.get()

        if select_all_state:
            info_listbox.select_set(0, tk.END)
        else:
            info_listbox.selection_clear(0, tk.END)

    def country_list_from_csv(file_path):
        countries = []
        with open('anthems.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                countries.append(row['Country'])
        return countries

    def filter_countries(search_query):
        # Filter countries based on the search query
        filtered_countries = [country for country in all_countries if search_query.lower() in country.lower()]
        return filtered_countries

    def update_listbox(search_query=""):
        info_listbox.delete(0, tk.END)  # Clear the current items

        filtered_countries = filter_countries(search_query)

        for country_search in filtered_countries:
            info_listbox.insert(tk.END, country_search)

    def search_entry_changed(event):
        if search_entry:
            # Update the listbox based on the search entry content
            search_query = search_entry.get()
            update_listbox(search_query)

    f1 = ctk.CTkFrame(master=compare_anthems, height=TotalHeight, width=TotalWidth, fg_color='white', corner_radius=0)
    f1.place(x=0, y=0)

    text_var = tk.StringVar(value="Comparison")

    l1 = ctk.CTkLabel(f1, textvariable=text_var, width=200, height=70, fg_color="#4473c5", corner_radius=15,
                      text_color='white', font=('italic', 22))
    l1.place(relx=0.5, rely=0.03, anchor='center')

    text_var1 = tk.StringVar(value="Select the countries you want to compare:")

    l2 = ctk.CTkLabel(f1, textvariable=text_var1, text_color="#4473c5", fg_color='white', font=('italic', 22))
    l2.place(relx=0.5, rely=0.2, anchor='center')

    # Entry for searching countries
    search_entry = Entry(f1, width=20, font=("Helvetica", 22))
    search_entry.place(relx=0.45, rely=0.3, anchor='center')
    search_entry.bind("<KeyRelease>", search_entry_changed)

    # Create a variable to track whether all countries are selected
    select_all_var = BooleanVar()
    select_all_var.set(False)

    # Checkbox for selecting all countries
    select_all_checkbox = ttk.Checkbutton(f1, text="Select All", variable=select_all_var, command=toggle_select_all,
                                          font=("Helvetica", 22), background='white', fg='black', bd=5)
    select_all_checkbox.pack(pady=10, side=tk.LEFT)
    select_all_checkbox.place(relx=0.58, rely=0.3, anchor='center')

    info_listbox = Listbox(f1, width=30, height=5, bg="lightblue", selectmode=tk.MULTIPLE, bd=5, relief=tk.RIDGE,
                           borderwidth=2, font=("Helvetica", 22))
    info_listbox.place(relx=0.5, rely=0.45, anchor='center')

    scrollbar = tk.Scrollbar(info_listbox, orient=tk.VERTICAL, command=info_listbox.yview)
    scrollbar.place(relx=0.98, rely=0.5, anchor='center', relheight=1)

    # Link Scrollbar to Listbox
    info_listbox.config(yscrollcommand=scrollbar.set)

    # Get the list of countries from the CSV file
    all_countries = country_list_from_csv('anthems.csv')

    # Insert countries into the info_listbox
    for country in all_countries:
        info_listbox.insert(tk.END, country)

    compare_button = ctk.CTkButton(f1, fg_color='#4473c5', font=('Helvetica', 22), text='Compare',
                                   command=lambda: on_select(None),
                                   height=50, width=200, text_color='white', corner_radius=10)
    compare_button.place(relx=0.5, rely=0.6, anchor='center')

    text_var2 = tk.StringVar(value="Generate anthem with compared words:")
    l3 = ctk.CTkLabel(f1, textvariable=text_var2, text_color="#4473c5", fg_color='white', font=('italic', 22))
    l3.place(relx=0.5, rely=0.68, anchor='center')

    # Use CTkButton instead of tk Button with rounded corners
    custom_button = ctk.CTkButton(f1, text="Generate", fg_color='#4473c5', text_color='white', height=50, width=200,
                                  command=lambda:
                                  Generate(None, 1),
                                  corner_radius=10, font=("Helvetica", 22))
    custom_button.place(relx=0.5, rely=0.76, anchor='center')

    bk_button = ctk.CTkButton(f1, fg_color='#4473c5', font=('Helvetica', 18), text='Back', text_color='white',
                              height=50, width=70, corner_radius=10, command=main)
    bk_button.place(relx=0.02, rely=0.022, anchor='center')


def Generate(my_entry, a):
    if a == 0:
        user_anthem = my_entry.get()
    global dfs
    ctk.CTkCanvas(win)
    screen = Canvas(master=win, bg='white', height=TotalHeight, width=TotalWidth)
    screen.place(x=0, y=0)
    generate = ctk.CTkFrame(master=screen, height=TotalHeight, width=TotalWidth, fg_color='white', corner_radius=0)
    generate.pack()

    f1 = ctk.CTkFrame(master=generate, height=TotalHeight, width=TotalWidth, fg_color='#4473c5', corner_radius=0)
    f1.place(x=0, y=0)
    # print(dfs)

    openai.api_key = ''

    def get_completion(text, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": text}]
        response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, )
        return response.choices[0].message["content"]

    if a == 0:
        prompt = user_anthem
    if a == 1:
        prompt = (
            f"genrate national anthem of lyrics 10 lines only from only words with highest value in given data\n{dfs} ")
    response_data = get_completion(prompt)
    print(response_data)

    text_var = tk.StringVar(value="Anthem")
    label = ctk.CTkLabel(f1, textvariable=text_var, width=200, height=70, text_color="#4473c5", corner_radius=15,
                         fg_color='white', font=('italic', 22))
    label.place(relx=0.5, rely=0.03, anchor='center')

    f2 = ctk.CTkFrame(master=f1, height=TotalHeight - TotalHeight / 3, width=TotalWidth - TotalWidth / 4,
                      fg_color='white', corner_radius=20)
    f2.place(relx=0.5, rely=0.5, anchor='center')
    l2 = ctk.CTkLabel(f2, text_color="#4473c5", corner_radius=15,
                      fg_color='white', font=('italic', 22))
    l2.place(relx=0.5, rely=0.5, anchor='center')
    l2.configure(text=response_data)

    bk_button = ctk.CTkButton(f1, text_color='#4473c5', font=('Helvetica', 18), text='Back', fg_color='white',
                              height=50, width=70, corner_radius=10, command=main)
    bk_button.place(relx=0.02, rely=0.02, anchor='center')


def Generate_Anthtem_Page():
    ctk.CTkCanvas(win)
    screen = Canvas(master=win, bg='white', height=TotalHeight, width=TotalWidth)
    screen.place(x=0, y=0)
    generate_anthems = ctk.CTkFrame(master=screen, height=TotalHeight, width=TotalWidth, fg_color='white',
                                    corner_radius=0)
    generate_anthems.pack()

    def input_words():
        input_w = my_entry.get()
        print(input_w)

    f1 = ctk.CTkFrame(master=generate_anthems, height=TotalHeight, width=TotalWidth, fg_color='white', corner_radius=0)
    f1.place(x=0, y=0)

    text_var = tk.StringVar(value="Generation")

    label = ctk.CTkLabel(f1, textvariable=text_var, width=200, height=70, fg_color="#4473c5", corner_radius=15,
                         text_color='white', font=('italic', 22))
    label.place(relx=0.5, rely=0.03, anchor='center')

    f2 = ctk.CTkFrame(f1, height=400, width=400, fg_color='#4473c5', corner_radius=20)
    f2.place(relx=0.5, rely=0.48, anchor='center')

    text_var1 = tk.StringVar(value="Enter the words with which you want to\n generate anthem :-)")

    label = ctk.CTkLabel(master=f2, textvariable=text_var1, height=70, text_color='#4473c5', font=('italic', 18),
                         fg_color='white', corner_radius=10)
    label.place(relx=0.5, rely=0.2, anchor='center')

    my_entry = ctk.CTkEntry(master=f2, fg_color='white', text_color='#4473c5', width=200, height=35, corner_radius=10)
    my_entry.place(relx=0.5, rely=0.5, anchor='center')
    # global var213123

    gnt_button = ctk.CTkButton(master=f2, fg_color='white', font=('italic', 16), text='Generate Anthem',
                               text_color='#4473c5', height=50, width=200, corner_radius=10,
                               command=lambda: Generate(my_entry, 0))
    gnt_button.place(relx=0.5, rely=0.8, anchor='center')

    bk_button = ctk.CTkButton(f1, fg_color='#4473c5', font=('Helvetica', 18), text='Back', text_color='white',
                              height=50, width=70, corner_radius=10, command=main)
    bk_button.place(relx=0.02, rely=0.02, anchor='center')


def main():
    ctk.CTkCanvas(win)
    screen = Canvas(win, bg='white', height=TotalHeight, width=TotalWidth)
    screen.place(x=0, y=0)
    main_page = ctk.CTkFrame(master=screen, height=TotalHeight, width=TotalWidth, fg_color='white', corner_radius=0)
    main_page.pack()

    f1 = ctk.CTkFrame(master=main_page, height=TotalHeight, width=TotalWidth / 2, fg_color='white', corner_radius=0)
    f1.pack(side='right')
    f2 = ctk.CTkFrame(master=main_page, height=TotalHeight, width=TotalWidth / 2, fg_color='white', corner_radius=0)
    f2.pack(side='left')

    f3 = ctk.CTkFrame(master=f1, height=TotalHeight / 3, width=TotalWidth / 2, fg_color='white', corner_radius=0)
    f3.place(x=0, y=0)
    f4 = ctk.CTkFrame(master=f1, height=TotalHeight / 2, width=TotalWidth / 2, fg_color='white', corner_radius=0)
    f4.place(x=0, y=TotalHeight / 3)

    l1 = ctk.CTkLabel(f3, text='LyricsVista', fg_color='white', text_color='#4473c5', width=TotalWidth / 2,
                      font=('Helvetica', 60))
    l1.place(relx=0.45, rely=0.7, anchor='center')

    b1 = ctk.CTkButton(master=f4, text='Compare\n Anthems', fg_color='#4473c5', hover_color='#AAA9FF', width=200,
                       height=200, font=('Helvetica', 30), command=CompareAnthems)
    b1.place(relx=0.18, rely=0.3, anchor='w')
    b2 = ctk.CTkButton(master=f4, text='Generate\n Anthems', fg_color='#4473c5', hover_color='#AAA9FF', width=200,
                       height=200, font=('Helvetica', 30), command=Generate_Anthtem_Page)
    b2.place(relx=0.82, rely=0.3, anchor='e')

    f5 = ctk.CTkFrame(master=f2, corner_radius=70, fg_color='#4473c5', height=TotalHeight, width=TotalWidth / 1.5)
    f5.place(x=-350, y=0)
    l2 = ctk.CTkLabel(f2,
                      text='A national anthem is not merely a musical composition; it becomes a sacred hymn, '
                           'a poetic ode to the resilience and unity of a nation. It whispers the echoes of the past, '
                           'sings the stories of its people, and orchestrates the dreams of a shared future, '
                           'all harmonized in the language of music.', 
                      fg_color='#4473c5', text_color='white', font=('Helvetica', 30), wraplength=TotalWidth / 4)
    l2.place(relx=0.45, rely=0.5, anchor='center')


win = ctk.CTk()
win.geometry("1550x1080")
win.title("LyricsVista")
win.config(background='white')
main()
win.mainloop()
