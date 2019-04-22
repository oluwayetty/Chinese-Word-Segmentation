
f1 = "/Users/oluwayetty1/Desktop/School/1-2/NLP/public_homework_1/resources/input.txt"
f2 = "/Users/oluwayetty1/Desktop/School/1-2/NLP/public_homework_1/resources/output.txt"
count = 0
with open(f1) as f1, open(f2) as f2:
         for f1_line,f2_line in zip(f1,f2): #this will return the same lines from both files
            if len(f1_line.rstrip()) != len(f2_line.rstrip()):
                count = count + 1
                import ipdb; ipdb.set_trace()
                print("f1 ----", f1_line.rstrip())
                print("f2 ----", f2_line.rstrip())
         print(count)
