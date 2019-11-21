from math import factorial


def combination(n, m):
    numerator = factorial(n)
    denominator = factorial(n - m)
    return numerator / denominator


def permutation(n, m):
    numerator = factorial(n)
    denominator = factorial(m) * factorial(n - m)
    return numerator / denominator


def dis_Ada_Bob():
    res = []
    for x in range(3, 11):
        prob = combination(7, x - 3) * combination(990, 10 - x) / combination(997, 7)
        res.append(prob)
        print('{:.2e}'.format(prob))
    return res


def dis_Ada_Cathy():
    print('------------------------------')
    res = []
    for x in range(1, 11):
        prob = combination(10, x) * combination(990, 10 - x) / combination(1000, 10)
        res.append(prob)
        print('{:.2e}'.format(prob))
    return res


if __name__ == '__main__':
    prob_a_b = dis_Ada_Bob()
    prob_a_c = dis_Ada_Cathy()

    # Euclidean
    res1 = 0
    for i in range(len(prob_a_b) - 1):
        temp = 0
        for j in range(i + 3, len(prob_a_c)):
            temp += prob_a_c[j]
        res1 += prob_a_b[i] * temp
    print('Euclidean: {:.2e}'.format(res1))

    # Jaccard
    res2 = 0
    for i in range(len(prob_a_b)):
        temp = 0
        for j in range(i + 3):
            temp += prob_a_c[j]
        res2 += prob_a_b[i] * temp
    print('Jaccard: {:.2e}'.format(res2))


