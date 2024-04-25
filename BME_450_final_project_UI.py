import torch
import torch.nn as nn
import tkinter as tk
from tkinter import *
from tkinter.font import Font
from datetime import datetime as dt
import joblib

def run_NN(values, text_boxes, age_days):
    inputs = [age_days]
    inputs.append(values[0]+1)
    inputs.append(round((int(text_boxes[3])*12 + int(text_boxes[4]))*2.54))
    inputs.append(round(int(text_boxes[5])/2.205))
    inputs.append(int(text_boxes[6]))
    inputs.append(int(text_boxes[7]))
    inputs.append(values[2]+2*values[3]+3*values[4])
    inputs.append(values[5]+2*values[6]+3*values[7])
    inputs.append(values[8])
    inputs.append(values[10])
    inputs.append(values[12])
    print(inputs)
    input_tensor = torch.tensor(inputs, dtype=torch.float32)
    input_tensor_shape = input_tensor.reshape(1, -1)
    scaled_input_tensor = torch.tensor(scaler.transform(input_tensor_shape), dtype=torch.float32)
    with torch.no_grad():
        output = model(scaled_input_tensor.unsqueeze(0))
        predicted_class = torch.argmax(output, dim=1).item()
        print(predicted_class)
    if predicted_class == 1:
        return('HIGH')
    else:
        return('LOW')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2), 
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

model.load_state_dict(torch.load('final_weights.pth'))
model.eval() 

current = dt.now()
cur_date = [current.month, current.day, current.year]

scaler = joblib.load('scaler.pkl')

root = Tk()
root.title("Heart Disease Predictor User Inputs")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 1000
window_height = 750
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

def on_entry_click(event, entry, prompt_text):
    if entry.get() == prompt_text:
        entry.delete(0, tk.END)  # Delete the current text in the entry widget
        entry.config(fg='black')  # Change text color to black

def on_focus_out(event, entry, prompt_text):
    if entry.get() == '':
        entry.insert(0, prompt_text)  # Insert the prompt text back into the entry widget
        entry.config(fg='grey')  # Change text color to grey

def on_root_click(event):
    event.widget.focus_set()

def on_type(event,entry):
    if entry['state'] != 'disabled':
        for k in range(3):
            auto_patient_res[k].set(0) 

def text_val(new_val,ind):
    try:
        if new_val == prompt_text[ind] or (new_val == '' or 0 <= int(new_val) <= entry_max[ind]):
            return True
        else:
            return False
    except ValueError:
        return False

def button_click(event, button, ind):
    if button['state'] != 'disabled':
        for k in range(3):
            auto_patient_res[k].set(0)
        if ind in button_pairs[0]:
            for k in button_pairs[0]: 
                if k != ind:
                    button_results[k].set(0)        
        elif ind in button_pairs[1]:
            for k in button_pairs[1]: 
                if k != ind:
                    button_results[k].set(0) 
        elif ind in button_pairs[2]:
            for k in button_pairs[2]: 
                if k != ind:
                    button_results[k].set(0)
        elif ind in button_pairs[3]:
            for k in button_pairs[3]: 
                if k != ind:
                    button_results[k].set(0)
        elif ind in button_pairs[4]:
            for k in button_pairs[4]: 
                if k != ind:
                    button_results[k].set(0)
        elif ind in button_pairs[5]:
            for k in button_pairs[5]: 
                if k != ind:
                    button_results[k].set(0)

def auto_button_click(event, button, ind):
    if button['state'] != 'disabled':
        entry_col_black(entry)
        for k in entry_pairs: 
            if k != ind:
                auto_patient_res[k].set(0)
        for k in range(14):
            button_results[k].set(auto_patient_buttons[ind][k])
        for k in range(8):
            entry_results[k].set(auto_patient_entries[ind][k])

def entry_col_grey(entry):
    for entry in root.winfo_children():
        if isinstance(entry, tk.Entry):
            entry.config(fg='grey')

def entry_col_black(entry):
    for entry in root.winfo_children():
        if isinstance(entry, tk.Entry):
            entry.config(fg='black')

def clear_click(event, button):
    if button['state'] != 'disabled':
        entry_col_grey(entry)
        for k in range(3):
            auto_patient_res[k].set(0)
        for k in range(14):
            button_results[k].set(0)
        for k in range(8):
            entry_results[k].set(prompt_text[k])

def submit_click(event, button):
    if button['state'] != 'disabled':
        values = [intvar.get() for intvar in button_results]
        text_boxes = [strvar.get() for strvar in entry_results]
        if text_boxes[0] == prompt_text[0] or text_boxes[1] == prompt_text[1] or text_boxes[2] == prompt_text[2] or text_boxes[3] == prompt_text[3] or text_boxes[4] == prompt_text[4] or text_boxes[5] == prompt_text[5] or text_boxes[6] == prompt_text[6] or text_boxes[7] == prompt_text[7]:
            update_error_message('Please fill all blanks')
        elif (values[0]==0 and values[1]==0) or (values[2]==0 and values[3]==0 and values[4]==0) or (values[5]==0 and values[6]==0 and values[7]==0) or (values[8]==0 and values[9]==0) or (values[10]==0 and values[11]==0) or (values[12]==0 and values[13]==0):
            update_error_message('Please select an answer to all questions')
        elif int(text_boxes[0]) == 0 or int(text_boxes[1]) == 0 or int(text_boxes[2]) == 0:
            update_error_message('Please input a valid date')
        elif int(text_boxes[3]) == 0 and int(text_boxes[4]) == 0:
            update_error_message('Please input a valid height')
        elif int(text_boxes[5]) == 0:
            update_error_message('Please input a valid weight')
        elif int(text_boxes[6]) == 0 or int(text_boxes[7]) == 0 or int(text_boxes[6]) < int(text_boxes[7]):
            update_error_message('Please input a valid blood pressure value')
        else:
            cur_day = dt(cur_date[2],cur_date[0],cur_date[1])
            birth_day = dt(int(text_boxes[2]),int(text_boxes[0]),int(text_boxes[1]))
            age_days = cur_day - birth_day
            age_days = age_days.days
            if age_days < 0:
                update_error_message('Please input a valid date')
            else:
                error_label.place_forget()
                print(values)
                print(text_boxes)
                print(age_days)
                NN_result = run_NN(values,text_boxes,age_days)
                result_message.config(text=NN_result)
                update_label_color()
                result_title.place(x=400,y=300)
                result_message.place(x=400,y=330)
                lock_buttons()
                redo_button.place(x=405,y=390)
                edit_button.place(x=475,y=390)

def update_error_message(message):
    error_message.set(message)
    error_label.place(x=350,y=650, width=300)

def update_label_color():
    if result_message.cget("text") == "LOW":
        result_message.config(fg="green")
    elif result_message.cget("text") == "HIGH":
        result_message.config(fg="red")

def lock_buttons():
    for widget in root.winfo_children():
        if isinstance(widget, tk.Checkbutton) or isinstance(widget, tk.Button) and widget is not redo_button and widget is not edit_button:
            widget['state'] = 'disabled'
        elif isinstance(widget, tk.Entry):
            widget['state'] = 'disabled'

def try_again(event,button):
    for widget in root.winfo_children():
        if isinstance(widget, tk.Checkbutton) or isinstance(widget, tk.Button):
            widget['state'] = 'normal'
        elif isinstance(widget, tk.Entry):
            widget['state'] = 'normal'
    clear_click(event,button)
    result_title.place_forget()
    result_message.place_forget()
    redo_button.place_forget()
    edit_button.place_forget()

def edit_resp(event,button):
    for widget in root.winfo_children():
        if isinstance(widget, tk.Checkbutton) or isinstance(widget, tk.Button):
            widget['state'] = 'normal'
        elif isinstance(widget, tk.Entry):
            widget['state'] = 'normal'   
    result_title.place_forget()
    result_message.place_forget()
    redo_button.place_forget() 
    edit_button.place_forget()
        
# LABELS
label_positions = [(15,20),(80,20),(145,20),(210,20),(275,20),(340,20),(405,20),(470,20),(535,20),(600,20)]
labels = [
    'What is your birth date?',
    'What is your sex?',
    'What is your height?',
    'What is your weight (lbs.)?',
    'What is your blood pressure (mmHg)?',
    'What is your cholesterol level?',
    'What is your glucose level?',
    'Do you smoke?',
    'Do you regularly consume alcohol?',
    'Do you exercise regularly?']

for i in range(10):
    label = tk.Label(root, text=labels[i], font=('Helvetica', 12), fg='black')
    label.place(x=label_positions[i][1], y=label_positions[i][0])

# ENTRIES
entry_width = [3, 3, 5, 3, 3, 6, 7, 7]
entry_height = 3
entry_positions = [(42,25),(42,66),(42,107),(172,25),(172,66),(237,25),(302,25),(302,102)]
entry_max = [12,31,cur_date[2],9,11,9999,300,200]
entry_results = [tk.StringVar() for _ in range(8)]

prompt_text = ['MM', 'DD', 'YYYY', 'ft.', 'in.', 'Weight','Systolic', 'Diastolic']

for i in range(8):
    entry = tk.Entry(root, width=entry_width[i], font=('Helvetica', 12), textvariable=entry_results[i], validate='key', validatecommand=(root.register(lambda new_val, ind=i: text_val(new_val,ind)),'%P'),fg='grey',justify='center')
    entry.insert(0, prompt_text[i])  # Insert the prompt text into the entry widget
    entry.bind('<FocusIn>', lambda event, entry=entry, prompt_text=prompt_text[i]: on_entry_click(event, entry, prompt_text))
    entry.bind('<FocusOut>', lambda event, entry=entry, prompt_text=prompt_text[i]: on_focus_out(event, entry, prompt_text))
    entry.bind('<Key>', lambda event, entry=entry: on_type(event,entry))
    entry.place(x=entry_positions[i][1], y=entry_positions[i][0])

# BUTTONS
button_positions = [(107,25),(107,70),(367,25),(367,87),(367,193),(432,25),(432,87),(432,193),(497,25),(497,65),(562,25),(562,65),(627,25),(627,65)]

button_text = ['Male','Female','Normal','Above Normal','Well Above Normal','Normal','Above Normal','Well Above Normal','Yes','No','Yes','No','Yes','No']
button_pairs = [(0,1),(2,3,4),(5,6,7),(8,9),(10,11),(12,13)]
button_results = [tk.IntVar() for _ in range(14)]

for i in range(14):
    button = tk.Checkbutton(root, text=button_text[i], variable=button_results[i], font=('Helvetica', 11), indicatoron=0, bd=2, bg='#a6a6a6', fg='black', relief=tk.GROOVE)
    button.place(x=button_positions[i][1], y=button_positions[i][0])
    button.bind('<Button-1>',lambda event, button=button, ind=i: button_click(event, button, ind))

# AUTO-FILL
auto_patients = ['Patient 1', 'Patient 2', 'Patient 3']
auto_patient_buttons = [(1,0,0,0,1,0,1,0,0,1,0,1,1,0),(0,1,0,1,0,0,1,0,1,0,1,0,1,0),(1,0,1,0,0,1,0,0,0,1,0,1,1,0)]
auto_patient_entries = [('01','15','1957','5','7','191','142','83'),('05','23','1983','5','4','135','113','76'),('06','09','2003','6','2','220','130','90')]
entry_pairs = [0, 1, 2]
auto_patient_prompt = 'Auto-Fill Example Data'
auto_buttons_pos = [(42,400),(42,470),(42,540)]
auto_patient_res = [tk.IntVar() for _ in range(3)]

auto_label = tk.Label(root, text=auto_patient_prompt, font=('Helvetica', 12), fg='black')
auto_label.place(x=421, y=15)

for i in range(3):
    auto_button = tk.Checkbutton(root, text=auto_patients[i], variable=auto_patient_res[i], font=('Helvetica', 11), indicatoron=0, bd=2, bg='#a6a6a6', fg='black', relief=tk.GROOVE)
    auto_button.place(x=auto_buttons_pos[i][1], y=auto_buttons_pos[i][0])
    auto_button.bind('<Button-1>',lambda event, button=auto_button, ind=i: auto_button_click(event, button, ind))

# CLEAR BUTTON
clear_button = tk.Button(root, text='Clear', font=('Helvetica', 12), bd=2, bg='#a6a6a6', fg='#c40d00', relief=tk.GROOVE)
clear_button.place(x=660,y=42,height=28)
clear_button.bind('<Button-1>',lambda event, button=clear_button: clear_click(event, button))

# ERRORS
error_message = tk.StringVar()
error_label = tk.Label(root, textvariable=error_message, font=('Helvetica', 9), fg='red')

# SUBMIT BUTTON
end_button = tk.Button(root, text='Submit', font=('Helvetica', 14), bd=2, bg='#a6a6a6', fg='black', relief=tk.GROOVE)
end_button.place(x=350,y=675, width=300)
end_button.bind('<Button-1>',lambda event, button=end_button: submit_click(event, button))

# RESULT
result_title = tk.Label(root, text='Risk of Heart Disease:', font=('Helvetica', 14), fg = 'black')
result_message = tk.Label(root, text='', font=('Helvetica', 24))

# TRY AGAIN
redo_button = tk.Button(root, text='Redo', font=('Helvetica', 14), bd=2, bg='#578feb', fg='black', relief=tk.GROOVE)
redo_button.bind('<Button-1>',lambda event, button=redo_button: try_again(event, button))

# EDIT RESPONSES
edit_button = tk.Button(root, text='Edit', font=('Helvetica', 14), bd=2, bg='#578feb', fg='black', relief=tk.GROOVE)
edit_button.bind('<Button-1>',lambda event, button=edit_button: edit_resp(event, button))

# e = Entry(root, width=20, borderwidth=5, bg="#5f80d9", fg="#ffffff", font=Font(family="Digital-7 Italic", size="22", weight="bold"))
# e.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

root.bind_all('<Button>', on_root_click)
root.mainloop()