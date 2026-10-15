"""Column-level lineage without publishing any customer records."""
import csv
from common import HERE,RAW,DERIVED,read,write,sha,now


def main():
    with (RAW/'feature_definitions.csv').open(encoding='utf-8-sig',newline='') as f:
        definitions={r['Variable']:r['Description'] for r in csv.DictReader(f)}
    rows=[]
    for variant in ['champion_lgb','champion_cat']:
        s=read(DERIVED/variant/'model_input/state.json')
        for kind,names in [('numeric',s['numeric_names']),('categorical',s['categorical_names'])]:
            for name in names:
                group,derived=name.split('__',1)
                source_name=derived.removesuffix('__missing')
                matches=[v for v in definitions if v in source_name]
                raw=matches[0] if len(matches)==1 else ''
                # Only a unique literal match inherits an official definition.
                description=definitions.get(raw,'')
                indicator=derived.endswith('__missing')
                st=s['numeric_states'].get(name.removesuffix('__missing'),{})
                rows.append(dict(variant=variant,input_name=name,source_table_group=group,
                            kind='missing_indicator' if indicator else kind,
                            literal_source_variable=raw,official_variable_description=description,
                            definition_status='UNIQUE_LITERAL_MATCH' if raw else 'NO_UNIQUE_DEFINITION_MATCH',
                            numeric_missing_rate_A=st.get('missing_p',''),
                            note='Official definition describes source variable, not automatically an aggregated economic construct'))
    path=HERE/'feature_catalog.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    write(HERE/'feature_catalog_receipt.json',dict(status='GENERATED',utc=now(),rows=len(rows),
          definition_file_sha256=sha(RAW/'feature_definitions.csv'),catalog_sha256=sha(path),contains_customer_records=False))
    print('CATALOG_READY',len(rows),'columns')


if __name__=='__main__':main()
