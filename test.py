from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

path = './main/cifar10/run2/cifar10-linear-1-4/'
main_event = EventAccumulator(path + 'main_acc')
aux_event_path = path + 'sub_acc_'
main_event.Reload()

acc = main_event.Scalars('Main_Accuracy')
main_acc = list()
for a in acc:
    main_acc.append(a.value)

sub_accs = list()
for i in range(9):
    sub_path = aux_event_path + str(i)
    sub_acc = list()
    event = EventAccumulator(sub_path)
    event.Reload()

    acc = event.Scalars('AccuracySubNetwork')
    for a in acc:
        sub_acc.append(a.value)

    sub_accs.append(sub_acc)

l, = plt.plot(main_acc, marker='o')
l.set_label('Main Accuracy')
markers = ['<', '2', 's', 'P', 'X', '|', '+', 'd', '*']
for i in range(9):
    l, = plt.plot(sub_accs[i], marker=markers[i])
    l.set_label('Sub Network {0}'.format(str(i + 1)))
plt.legend()
plt.grid(True)
plt.show()
