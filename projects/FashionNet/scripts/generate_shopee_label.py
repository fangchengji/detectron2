import os
import glob
import json
from PIL import Image
import itertools

root_dir = '/Users/fangcheng.ji/Documents/datasets/shopee_fashion/train_result'
anno_file = os.path.join(root_dir, 'shopee_train_6k.json')


classes = ['commodity', 'model', 'detail', 'specification', 'unknown']
parts = ['top', 'down', 'whole']


def create_category2():
    categories2 = []
    categories2.append({
        'id': 1,
        'name': "commodity",
        'supercategory': "fashion",
    })
    categories2.append({
        'id': 2,
        'name': "model",
        'supercategory': "fashion"
    })
    categories2.append({
        'id': 3,
        'name': "detail",
        'supercategory': "fashion"
    })
    categories2.append({
        'id': 4,
        'name': "specification",
        'supercategory': "fashion"
    })
    categories2.append({
        'id': 5,
        'name': "unknown",
        'supercategory': "fashion"
    })

    return categories2

def create_category():
    dataset = {'categories': []}
    dataset['categories'].append({
        'id': 1,
        'name': "short_sleeved_shirt",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 2,
        'name': "long_sleeved_shirt",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 3,
        'name': "short_sleeved_outwear",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 4,
        'name': "long_sleeved_outwear",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 5,
        'name': "vest",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 6,
        'name': "sling",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 7,
        'name': "shorts",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 8,
        'name': "trousers",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 9,
        'name': "skirt",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 10,
        'name': "short_sleeved_dress",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 11,
        'name': "long_sleeved_dress",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 12,
        'name': "vest_dress",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    dataset['categories'].append({
        'id': 13,
        'name': "sling_dress",
        'supercategory': "clothes",
        'keypoints': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                      '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
                      '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
                      '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
                      '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82',
                      '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98',
                      '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112',
                      '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126',
                      '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                      '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154',
                      '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
                      '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182',
                      '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196',
                      '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                      '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224',
                      '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238',
                      '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252',
                      '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266',
                      '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                      '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294'],
        'skeleton': []
    })
    return dataset['categories']


def create_images(image_list):
    images = []
    for image_path in image_list:
        image = Image.open(image_path)
        w, h = image.size
        image_name = os.path.basename(image_path)
        images.append({
            "coco_url": "",
            "date_captured": "",
            "file_name": image_name,
            "flickr_url": "",
            "id": int(image_name.split('.')[0]),
            "license": 0,
            "width": w,
            "height": h
        })
    return images


def create_annotations2(image_list):
    annotations2 = []
    cls_dict = {key: i+1 for i, key in enumerate(classes)}
    pa_dict = {key: i+1 for i, key in enumerate(parts)}
    for image_path in image_list:
        sp = image_path.split('/')
        if sp[-2] in cls_dict:
            cat_id = cls_dict[sp[-2]]
        elif sp[-3] in cls_dict:
            cat_id = cls_dict[sp[-3]]
        else:
            raise Exception("Path error!")

        # model
        part_id = 0
        if cat_id == 2:
            part_id = pa_dict[sp[-2]]

        annotations2.append({
            'image_id': int(os.path.basename(image_path).split('.')[0]),
            'id': int(os.path.basename(image_path).split('.')[0]),
            'category2_id': cat_id,
            'part': part_id,
            # if toward is 0, it will be ignored when training
            'toward': 0
        })
    return annotations2


if __name__ == '__main__':
    unknown_paths = glob.glob(os.path.join(root_dir, 'unknown') + '/*.jpg')
    specification_paths = glob.glob(os.path.join(root_dir, 'specification') + '/*.jpg')
    detail_paths = glob.glob(os.path.join(root_dir, 'detail') + '/*.jpg')
    commodity_paths = glob.glob(os.path.join(root_dir, 'commodity') + '/*.jpg')
    model_paths = glob.glob(os.path.join(root_dir, 'model') + '/*/*.jpg')

    image_list = list(itertools.chain(unknown_paths,
                                      specification_paths,
                                      detail_paths,
                                      commodity_paths,
                                      model_paths))
    print(len(image_list))

    dataset = {
        "info": {},
        "licenses": [],
        "images": create_images(image_list),
        "annotations": [],
        "annotations2": create_annotations2(image_list),
        "categories": create_category(),
        "categories2": create_category2()
    }

    # # write to new json file
    with open(anno_file, 'w') as ff:
        json.dump(dataset, ff)

