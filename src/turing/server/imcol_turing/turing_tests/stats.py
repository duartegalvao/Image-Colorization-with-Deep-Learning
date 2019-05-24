from .models import TuringTest


def generate_statistics():
    print("===== UNET =====")
    generate_statistics_for_set(1)
    print("===== GAN =====")
    generate_statistics_for_set(2)
    print("===== VAC+GAN =====")
    generate_statistics_for_set(3)


def generate_statistics_for_set(set):
    data = TuringTest.objects.filter(set=set)

    correct_n = data.filter(is_correct=True).count()
    incorrect_n = data.filter(is_correct=False).count()
    success_rate = correct_n / (correct_n + incorrect_n)

    #time_mean = np.mean(ds['time'])
    #time_std = np.std(ds['time'])

    print("{} correct guesses, {} incorrect.".format(correct_n, incorrect_n))
    print("Success rate: {}%".format(success_rate * 100))
    #print("Mean time: {} ± {} ms".format(time_mean, time_std))
