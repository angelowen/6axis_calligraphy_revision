import os, re
from post_utils import argument_setting

def verification(args):

    flag = True
    for root, _, files in os.walk(args.save_path):
        # get the list of the csv file name
        csv_files = list(filter(lambda x: re.match(r'test_all_target.txt', x), files))
        # no test_all_target.txt in root
        if len(csv_files) == 0:
            # print(f'{root}:\tNo files found!!!')
            continue
        print(f'{root}\t{csv_files[0]}')

        # in test_all/
        if root[-4:] == '_all':
            # print('\tskip test_all/...')
            continue
        char_num = os.path.abspath(root)[-3:]
        if char_num.isdigit() == False:
            char_num = f'{args.test_char:03d}'
        with open(f'{args.input_path}/char00{char_num}_stroke.txt', mode='r') as correct_file:
            correct_content = correct_file.read()
            with open(f'{root}/{csv_files[0]}') as test_file:
                test_content = test_file.read()
                if correct_content != test_content:
                    print(f'\nError: {char_num} is NOT Correct!!!\n')
                    flag = False
                    continue
    if flag is True:
        print('\nVerification All Correct!!!')

if __name__ == '__main__':
    args = argument_setting()
    verification(args)