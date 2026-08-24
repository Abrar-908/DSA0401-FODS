login_attempts = int(input("Enter login attempts: "))
exam_time = int(input("Enter exam duration in minutes: "))

if login_attempts > 3 or exam_time > 180:
    print("Suspicious Activity - Alert")
else:
    print("Normal Activity")

print("Fraud Detection Report Generated")