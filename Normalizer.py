    #cap_shape = ('b', 'c', 'x', 'f', 'k', 's')
    #cap_surface = ('f', 'g', 'y', 's')
    #cap_color = ('n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y')
    #bruises = ('t', 'f')
    #odor = ('a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's')
    #gill_attachment = ('a', 'd', 'f', 'n')
    #gill_spacing = ('c', 'w', 'd')
    #gill_size = ('b', 'n')
    #gill_color = ('k', 'n', 'b', 'h', 'g', 'r', 'o' ,'p', 'u', 'e', 'w', 'y')
    #stalk_shape = ('e', 't')
    #stalk_root = ('b', 'c', 'u', 'e', 'z', 'r' '?')
    #stalk_surface_abv = ('f', 'y', 'k', 's',)
    #stalk_surface_blw = ('f', 'y', 'k', 's',)
    #stalk_color_abv = ('n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y')
    #stalk_color_blw = ('n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y')
    #veil_type = ('p', 'u')
    #veil_color = ('n', 'o', 'w', 'y')
    #ring_number = ('n', 'o', 't')
    #ring_type = ('c', 'e', 'f', 'l', 'n', 'p', 's', 'z')
    #spore_color = ('k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y')
    #population = ('a', 'c', 'n', 's', 'v', 'y')
    #habitat = ('g', 'l', 'm', 'p', 'u', 'w', 'd')
from decimal import *

class Normalizer(object):
    """Normalizes letter characteristics into 1-of-C binary encodings"""
    attribute_map = []
    attribute_map.append(('p', 'e'))
    attribute_map.append(('b', 'c', 'x', 'f', 'k', 's'))
    attribute_map.append(('f', 'g', 'y', 's'))
    attribute_map.append(('n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'))
    attribute_map.append(('t', 'f'))
    attribute_map.append(('a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'))
    attribute_map.append(('a', 'd', 'f', 'n'))
    attribute_map.append(('c', 'w', 'd'))
    attribute_map.append(('b', 'n'))
    attribute_map.append(('k', 'n', 'b', 'h', 'g', 'r', 'o' ,'p', 'u', 'e', 'w', 'y'))
    attribute_map.append(('e', 't'))
    attribute_map.append(('b', 'c', 'u', 'e', 'z', 'r', '?'))
    attribute_map.append(('f', 'y', 'k', 's',))
    attribute_map.append(('f', 'y', 'k', 's',))
    attribute_map.append(('n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'))
    attribute_map.append(('n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'))
    attribute_map.append(('p', 'u'))
    attribute_map.append(('n', 'o', 'w', 'y'))
    attribute_map.append(('n', 'o', 't'))
    attribute_map.append(('c', 'e', 'f', 'l', 'n', 'p', 's', 'z'))
    attribute_map.append(('k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'))
    attribute_map.append(('a', 'c', 'n', 's', 'v', 'y'))
    attribute_map.append(('g', 'l', 'm', 'p', 'u', 'w', 'd'))
    def __init__(self, data):
        self.data_set = data
        coding_length = 0
        for line in self.attribute_map:
            coding_length += len(line)
        coding_length = 22
        self.normalized = ([[0 for j in range(coding_length)] for i in range(len(self.data_set))]) #initialize empty binary data set

    def encode(self):
        getcontext().prec = 1;
        for line_number, line in enumerate(self.data_set):
            current_binary = []
            for index, data in enumerate(line):
                if index != 0:
                    #coding = self.get_coding(data, self.attribute_map[index])
                    current_binary.append(self.get_coding(data, self.attribute_map[index]))
                else:
                    if data == 'p':
                        final_prediction = (1, 0)
                    else:
                        final_prediction = (0, 1)
            current_binary.extend(final_prediction)
            self.normalized[line_number] = current_binary
        return self.normalized

    def get_coding(self, data, attr):
        for i, lt in enumerate(attr):
            if lt == data:
                return i+1 #Decimal(i)/Decimal(10) + Decimal(0.1);
    def encode2(self):
        for line_number, line in enumerate(self.data_set):
            current_binary = []
            for index, data in enumerate(line):
                if index != 0:
                    coding = self.get_binary(data, self.attribute_map[index])
                    current_binary.extend(coding)
                else:
                    if data == 'p':
                        final_prediction = (1, 0)
                    else:
                        final_prediction = (0, 1)
                    #final_prediction = self.get_binary(data, self.attribute_map[index])

            current_binary.extend(final_prediction)
            self.normalized[line_number] = current_binary
        return self.normalized

    def get_binary(self, letter, attribute):
        binary = []
        for index, lt in enumerate(attribute):
            if index == len(attribute) - 1:
                new_binary = []
                for i in attribute:
                    new_binary.append(-1)
                return new_binary
            elif letter == lt:
                binary.append(1)
                for i in range(index, len(attribute) - 1):
                    binary.append(0)
                return binary
            else:
                binary.append(0)
        return binary


