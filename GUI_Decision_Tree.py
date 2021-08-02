import pandas as pd
import tkinter.filedialog
from tkinter import *
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from PIL import Image, ImageTk


# THIS FUNCTION IS FOR UPLOADING DATASET AND DISPLAYING IT
def import_csv_data():
    global df
    global check
    filepath = tkinter.filedialog.askopenfilename()
    check = StringVar()
    file = open(filepath, 'r')
    t_upload.insert(END, filepath)
    df = pd.read_csv(filepath)
    check = "Dataset is uploaded"
    t_train.configure(background="#00ff00", foreground="#000000", font=("Courier", 12))
    t_train.insert(END, check)
    head_label = Label(main_window, text=df, height=20, width=50, font=("Courier", 12))
    head_label.configure(background="#ffad33")
    head_label.place(x="900", y="50")
    # print(df)

# THIS FUNCTION IS FOR TRAINING THE MODEL CALLED FROM TRAIN_MODEL
def train(attribute_selection):
    global decisionTree
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    decisionTree = DecisionTreeClassifier(criterion=attribute_selection, max_depth=4)
    decisionTree.fit(X_train, y_train)
    y_pred = decisionTree.predict(x_test)
    global accuracy
    global classification

    global confusion
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification = classification_report(y_test, y_pred)

# THIS FUNCTION IS CALLED WHEN TRAIN BUTTON IS CLICKED
def train_model(attribute_selection):
    if not len(t_train.get("1.0", 'end-1c')):
        error_message = "Upload the Dataset First!"
        t_train.delete("1.0", "end")
        t_train.configure(background="#ae0000", foreground="#ffffff", font=("Courier", 12))
        t_train.insert(END, error_message)
        return
    if attribute_selection == 'Gini':
        attribute_selection = "gini"
    else:
        attribute_selection = "entropy"
    train(attribute_selection)
    fig = plt.figure(figsize=(20, 10))
    tree_img = tree.plot_tree(decisionTree, filled=True)
    fig.savefig("decisionTree.png")
    check = "Training the model is done!"
    # print(check)
    t_train.configure(background="#00ff00", foreground="#000000", font=("Courier", 12))
    t_train.delete("1.0", "end")
    t_train.insert(END, check)

# THIS IS TO SHOW THE METRICS OF THE MODEL IN ANOTHER WINDOW
def predict_model():
    if not len(t_train.get("1.0", 'end-1c')):
        error_message = "Train the Model First!"
        t_train.delete("1.0", "end")
        t_train.configure(background="#ae0000", foreground="#ffffff", font=("Courier", 12))
        t_train.insert(END, error_message)
        return
    predict_window = Toplevel()
    predict_window.geometry('1200x800')
    predict_window.configure(background="#99ffff")
    predict_window.title('Accuracy')
    accuracy_text = Label(predict_window, text="Accuracy of the model is: ", font=("Arial", 20))
    accuracy_text.configure(foreground="#0000ff", background="#99ffff")
    accuracy_text.place(x="10", y="10")
    accuracy_label = Label(predict_window, text=accuracy, font=("Arial", 20))
    accuracy_label.configure(background="#99ffff")
    accuracy_label.place(x="400", y="10")
    confusion_text = Label(predict_window, text="Confusion Matrix is: ", font=("Arial", 20))
    confusion_text.configure(foreground="#0000ff", background="#99ffff")
    confusion_text.place(x="10", y="80")
    confusion_label = Label(predict_window, text=confusion, font=("Arial", 20))
    confusion_label.configure(background="#99ffff")
    confusion_label.place(x="200", y="150")
    classification_text = Label(predict_window, text="Classification Report is: ", font=("Arial", 20))
    classification_text.configure(foreground="#0000ff", background="#99ffff")
    classification_text.place(x="10", y="300")
    classification_label = Label(predict_window, text=classification, font=("Arial", 20))
    classification_label.configure(background="#99ffff")
    classification_label.place(x="300", y="350")
    predict_window.mainloop()
    # predict_window.destroy

# THIS FUNCTION IS USED TO PLOT THE DECISION TREE
def tree_plot():
    if not len(t_train.get("1.0", 'end-1c')):
        error_message = "Train the Model!"
        t_train.delete("1.0", "end")
        t_train.configure(background="#ae0000", foreground="#ffffff", font=("Courier", 12))
        t_train.insert(END, error_message)
        return
    tree_window = Toplevel()
    tree_window.geometry('1700x1500')
    tree_window.title('Decision Tree')
    image = Image.open("decisionTree.png")
    photo = ImageTk.PhotoImage(image)
    w = Label(tree_window, image=photo)
    w.configure()
    w.pack()
    tree_window.mainloop()


# MAKING A MAIN WINDOW WIDGET
main_window = tkinter.Tk()
main_window.geometry('1500x700')
main_window.title('Diabetes Predictions')
main_window.configure(background="#ffffcc")

browse_label = Label(main_window, text="Path of the Dataset: ", font=("Courier", 10))
browse_label.configure(background="#ffffcc")
browse_label.place(x="10", y="48")
# Dropdown menu options
options = [
    "Gini",
    "Entropy"
]
option_select = Label(main_window, text="Select the attribute selection method: ", font=("Courier", 10))
option_select.configure(background="#ffffcc")
option_select.place(x="10", y="90")
clicked = StringVar()   # datatype of menu text
clicked.set("Gini")     # initial menu text
drop = OptionMenu(main_window, clicked, *options)
drop.pack()     # Create Dropdown menu
drop.place(x="330", y="90")

t_upload = Text(main_window, width=54, height=2)
t_train = Text(main_window, width=60, height=3)

t_upload.place(x=180, y=46)
t_train.place(x=200, y=170)

browse_data = tkinter.Button(main_window, text='Browse Data Set', command=import_csv_data, height="2")
browse_data.place(x="640", y="46")
train_button = tkinter.Button(main_window, text='Train', command=lambda: train_model(clicked.get()), height="2", width="7", font=(4))
train_button.place(x="250", y="250")
predict_button = tkinter.Button(main_window, text='Predict', command=predict_model, height="2", width="7", font=(4))
predict_button.place(x="340", y="250")
tree_button = tkinter.Button(main_window, text='Show Tree', command=tree_plot, height="2", width="9", font=(4))
tree_button.place(x="430", y="250")
close_button = tkinter.Button(main_window, text='Close', command=main_window.destroy, height="2", width="7", font=(4))
close_button.place(x="545", y="250")


main_window.mainloop()
