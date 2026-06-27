WITH prod AS (
SELECT
*
FROM siebel.s_1st_of_val@siebel
WHERE active_flg = 'Y'
AND type = 'JET_SR_PRODUCT')
, theme AS (
SELECT
*
FROM siebel.s_lst_of_val@siebel
WHERE active_flg = 'Y'
AND type = 'JET_SR _THEME')
, category AS (
SELECT
*
FROM siebel.s_lst_of_val@siebel
WHERE active_flg = 'Y'
AND type = 'JET_SR_CATEGORY'
)
SELECT
ca.claim_num, 
ca.created,
cea.description_claim,
sr.act_close_dt,
pr.desc_text as product, 
th.desc_text as theme, 
cat.desc_text as category, 
ca.resp_dir, 
sr.sr_stat_id
FROM siebel.cx_claim_attr_x@siebel ca
LEFT JOIN siebel.s_srv_req@siebel
ON ca.row_id=sr.row_id
LEFT JOIN siebel.cx_clm_at_ext_x@siebel cea
ON cea.par_row_id=ca.row_id
LEFT JOIN prod pr
ON ca.sr_product = pr.name
LEFT JOIN theme th
ON ca.sr_theme = th. name
AND pr. row_id = th. sub_type
LEFT JOIN category cat
ON ca.sr_category = cat.name
AND th. row_id = cat.sub_type
WHERE sr.sr_stat_id != 'Отменено'
AND ca.resp_dir IN ('ДирПР', 'ДирПРПК')
AND ca.created >= to_date('{start_date}', 'YYYY-MM-DD')
AND ca.created < to_date('{end_date}', 'YYYY-MM-DD')