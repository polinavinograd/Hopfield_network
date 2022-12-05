import pickle
from random import choice

from matplotlib import image as mpimg

from Hopfield_network import HopfieldNetwork
from config import weights_path, path_pngs
from utils import view, create_input, create_pic, distort

flag = True
while flag:
    cho = int(input("1 - обучить, 2 - проверить образ, 3 - проверить испорченный образ, "
                    "4 - исказить образ и сохранить, 5 - загрузить образ, 0 - exit\n"))
    match cho:
        case 0:
            flag = 0
        case 1:
            N = HopfieldNetwork()
            N.train()

        case 2:
            N = HopfieldNetwork(weights_path)
            pic = mpimg.imread(f"{path_pngs}{choice(range(10))}.png")
            vec = create_input(pic)
            out_vec = N.test(vec)
            out_pic = create_pic(out_vec)
            view(pic, out_pic)

        case 3:
            distortion_rate = float(input("Введите степень искажённости картинки: "))
            N = HopfieldNetwork(weights_path)
            pic = mpimg.imread(f"{path_pngs}{choice(range(10))}.png")
            vec = create_input(pic)
            distort_vec = distort(vec, distortion_rate)
            distort_pic = create_pic(distort_vec)
            out_vec = N.test(distort_vec)
            out_pic = create_pic(out_vec)
            view(distort_pic, out_pic)

        case 4:
            distortion_rate = float(input("Введите степень искажённости картинки: "))
            pic_num = choice(range(10))
            pic = mpimg.imread(f"{path_pngs}{pic_num}.png")
            vec = create_input(pic)
            distort_vec = distort(vec, distortion_rate)
            with open(f"{pic_num}_{int(distortion_rate * 100)}.pickle", 'wb') as f:
                pickle.dump(distort_vec, f)

        case 5:
            file = input("Введите имя файла без расширения: ")
            with open(f"{file}.pickle", 'rb') as f:
                distort_vec = pickle.load(f)
            distort_pic = create_pic(distort_vec)
            N = HopfieldNetwork(weights_path)
            out_vec = N.test(distort_vec)
            out_pic = create_pic(out_vec)
            view(distort_pic, out_pic)

        case _:
            print("Error")
